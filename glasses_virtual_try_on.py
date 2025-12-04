import os, time, math, threading
import cv2, numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory
import mediapipe as mp
from typing import Dict, Tuple, Optional
from camera_async import AsyncVideoCapture

class TempleSkinningCache:
  
    def __init__(self):
        self.cache = {}  # {temple_id: temple_info}
    
    def get_or_compute_temple_info(self, temple_id, temple_rgba, pivot_src):
        if temple_id in self.cache:
            return self.cache[temple_id]
        
        # Calculate and cache
        info = self._compute_temple_info(temple_rgba, pivot_src)
        self.cache[temple_id] = info
        print(f"  [CACHE] Cache temple information: {temple_id}")
        return info
    
    def _compute_temple_info(self, temple_rgba, pivot_src):
        h, w = temple_rgba.shape[:2]
        alpha = temple_rgba[..., 3]
        
        # Endpoint Detection
        (x0, y0), (x1, y1) = pivot_src
        front_point = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
        
        ys, xs = np.where(alpha > 128)
        if len(xs) == 0:
            back_point = np.array([w - 10, front_point[1]], dtype=np.float32)
        else:
            distances = np.sqrt((xs - front_point[0])**2 + (ys - front_point[1])**2)
            farthest_idx = np.argmax(distances)
            back_point = np.array([float(xs[farthest_idx]), float(ys[farthest_idx])], 
                                 dtype=np.float32)
        
        # Width Measurement
        src_vec = back_point - front_point
        src_len = np.linalg.norm(src_vec) + 1e-9
        src_vec_norm = src_vec / src_len
        src_perp_norm = np.array([-src_vec_norm[1], src_vec_norm[0]], dtype=np.float32)
        
        widths = []
        n_samples = 5  
        for i in range(n_samples):
            t = i / (n_samples - 1) if n_samples > 1 else 0.5
            sample_point = front_point + src_vec * t
            
            for offset in range(1, 30):  
                pos_point = sample_point + src_perp_norm * offset
                neg_point = sample_point - src_perp_norm * offset
                
                x_pos, y_pos = int(pos_point[0]), int(pos_point[1])
                x_neg, y_neg = int(neg_point[0]), int(neg_point[1])
                
                has_pos = (0 <= x_pos < w and 0 <= y_pos < h and alpha[y_pos, x_pos] > 128)
                has_neg = (0 <= x_neg < w and 0 <= y_neg < h and alpha[y_neg, x_neg] > 128)
                
                if has_pos or has_neg:
                    widths.append(offset * 2)
                    break
        
        src_width = np.median(widths) if widths else 20.0
        
        return {
            'front_point': front_point,
            'back_point': back_point,
            'src_width': src_width,
            'src_vec_norm': src_vec_norm,
            'src_perp_norm': src_perp_norm
        }
    
    def clear_cache(self):
        self.cache.clear()
        print("  [CACHE] The temple cache has been cleared")


# Create a global cache object
_temple_cache = TempleSkinningCache()


def detect_temple_endpoints_auto(temple_rgba, pivot_src):
    h, w = temple_rgba.shape[:2]
    alpha = temple_rgba[..., 3]
    
    # Front End Point: Pivot Point Center
    (x0, y0), (x1, y1) = pivot_src
    front_point = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
    
    # Back end point: Find the farthest non-transparent pixel as the anchor point
    ys, xs = np.where(alpha > 128)
    if len(xs) == 0:
        back_point = np.array([w - 10, front_point[1]], dtype=np.float32)
        return front_point, back_point
    
    distances = np.sqrt((xs - front_point[0])**2 + (ys - front_point[1])**2)
    farthest_idx = np.argmax(distances)
    back_point = np.array([float(xs[farthest_idx]), float(ys[farthest_idx])], dtype=np.float32)
    
    temple_length = np.linalg.norm(back_point - front_point)
    print(f"  [TEMPLE] Front:({front_point[0]:.1f},{front_point[1]:.1f}) "
          f"Back:({back_point[0]:.1f},{back_point[1]:.1f}) Len:{temple_length:.1f}px")
    
    return front_point, back_point


def warp_temple_with_skinning(temple_rgba, src_front, src_back, 
                              dst_front, dst_back, out_shape, width_scale=1.0):

    src_front = np.array(src_front, dtype=np.float32).flatten()
    src_back = np.array(src_back, dtype=np.float32).flatten()
    dst_front = np.array(dst_front, dtype=np.float32).flatten()
    dst_back = np.array(dst_back, dtype=np.float32).flatten()
    

    h, w = temple_rgba.shape[:2]
    alpha = temple_rgba[..., 3]
    
    # Calculate the average width of the eyeglass temple (vertical to the temple direction)
    src_vec = src_back - src_front
    src_len = np.linalg.norm(src_vec) + 1e-9
    src_vec_norm = src_vec / src_len
    
    # perpendicular to the temple direction
    src_perp_norm = np.array([-src_vec_norm[1], src_vec_norm[0]], dtype=np.float32)
    
    # Take multiple cross-sectional samples along the arm of the mirror and measure the width
    widths = []
    n_samples = 10  
    for i in range(n_samples):
        t = i / (n_samples - 1) if n_samples > 1 else 0.5
        sample_point = src_front + src_vec * t
        
        for offset in range(1, 50):  
            pos_point = sample_point + src_perp_norm * offset
            neg_point = sample_point - src_perp_norm * offset
            
            x_pos, y_pos = int(pos_point[0]), int(pos_point[1])
            x_neg, y_neg = int(neg_point[0]), int(neg_point[1])
            
            has_pos = (0 <= x_pos < w and 0 <= y_pos < h and alpha[y_pos, x_pos] > 128)
            has_neg = (0 <= x_neg < w and 0 <= y_neg < h and alpha[y_neg, x_neg] > 128)
            
            if has_pos or has_neg:
                widths.append(offset * 2)
                break
    
    if widths:
        src_width = np.median(widths)
    else:
        src_width = 20.0  
    
    # Calculate the third point (vertical direction) with apply width scaling
    src_third = src_front + src_perp_norm * src_width
    
    dst_vec = dst_back - dst_front
    dst_perp = np.array([-dst_vec[1], dst_vec[0]], dtype=np.float32)
    dst_perp_norm = dst_perp / (np.linalg.norm(dst_perp) + 1e-9)
    
  
    dst_width = src_width * width_scale
    dst_third = dst_front + dst_perp_norm * dst_width
    
    
    src_pts = np.array([src_front, src_back, src_third], dtype=np.float32)
    dst_pts = np.array([dst_front, dst_back, dst_third], dtype=np.float32)
    
    # Computing Affine Transformations
    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    # Apply The Transformation
    H_out, W_out = out_shape[:2]
    warped = cv2.warpAffine(temple_rgba, M, (W_out, H_out),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))
    
    bgr = warped[..., :3].astype(np.float32)
    alpha = warped[..., 3:4].astype(np.float32) / 255.0
    
    return bgr, alpha


def warp_temple_with_skinning_fast(temple_rgba, temple_info, 
                                   dst_front, dst_back, out_shape, width_scale=1.0):
    # Retrieve information from the cache
    src_front = temple_info['front_point']
    src_back = temple_info['back_point']
    src_width = temple_info['src_width']
    src_perp_norm = temple_info['src_perp_norm']
    
    dst_front = np.array(dst_front, dtype=np.float32).flatten()
    dst_back = np.array(dst_back, dtype=np.float32).flatten()
    
    # Calculate the third point of the source 
    src_third = src_front + src_perp_norm * src_width
    
    dst_vec = dst_back - dst_front
    dst_perp = np.array([-dst_vec[1], dst_vec[0]], dtype=np.float32)
    dst_perp_norm = dst_perp / (np.linalg.norm(dst_perp) + 1e-9)
    dst_width = src_width * width_scale
    dst_third = dst_front + dst_perp_norm * dst_width
    
    src_pts = np.array([src_front, src_back, src_third], dtype=np.float32)
    dst_pts = np.array([dst_front, dst_back, dst_third], dtype=np.float32)
    
    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    H_out, W_out = out_shape[:2]
    warped = cv2.warpAffine(temple_rgba, M, (W_out, H_out),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))
    
    bgr = warped[..., :3].astype(np.float32)
    alpha = warped[..., 3:4].astype(np.float32) / 255.0
    
    return bgr, alpha


# Database Path（set your own database path）
FACE_DB_PATH = r"D:\intern\web_ar_virtual_try_on\database\face_db\face_by_shape.csv"
EYE_DB_PATH = r"D:\intern\web_ar_virtual_try_on\database\eye_db\eyes_by_shape.csv"


def apply_temple_fadeout(temple_alpha, hinge_point, anchor_point, 
                         fade_start_ratio, fade_end_ratio, frame_shape):
    h, w = frame_shape[:2]
    
    # Calculate the direction and length of the eyeglass temples
    arm_vec = anchor_point - hinge_point
    arm_length = np.linalg.norm(arm_vec) + 1e-9
    arm_dir = arm_vec / arm_length
    
    # Create a coordinate grid
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    # Calculate the projection distance of each pixel along the mirror arm direction
    # Vector relative to the hinge point
    px_vec_x = x_grid - hinge_point[0]
    px_vec_y = y_grid - hinge_point[1]
    
    # Projection in the direction of the eyeglass temple arms
    proj_dist = px_vec_x * arm_dir[0] + px_vec_y * arm_dir[1]
    
    # Calculate the fade coefficient
    fade_start_dist = arm_length * fade_start_ratio
    fade_end_dist = arm_length * fade_end_ratio
    

    fade_factor = np.clip((fade_end_dist - proj_dist) / (fade_end_dist - fade_start_dist + 1e-9), 0, 1)
    
    
    
    if temple_alpha.ndim == 3:
        fade_factor = fade_factor[..., np.newaxis]  
    
    return temple_alpha * fade_factor


def apply_temple_fadeout_fast(temple_alpha, hinge_point, anchor_point, 
                              fade_start_ratio=0.85, fade_end_ratio=1.15):

    if temple_alpha.ndim == 3:
        temple_alpha_2d = temple_alpha.squeeze()
    else:
        temple_alpha_2d = temple_alpha
    
    h, w = temple_alpha_2d.shape
    
    if fade_start_ratio >= fade_end_ratio:
        return temple_alpha
    
    # Calculate the direction and length of the eyeglass temples
    arm_vec = anchor_point - hinge_point
    arm_length = np.linalg.norm(arm_vec)
    
    if arm_length < 1e-6:
        return temple_alpha
    
    arm_dir = arm_vec / arm_length
    
    # Use meshgrid instead of mgrid to create coordinate grids
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    px_vec_x = x_grid - hinge_point[0]
    px_vec_y = y_grid - hinge_point[1]
    

    proj_dist = px_vec_x * arm_dir[0] + px_vec_y * arm_dir[1]
    
 
    fade_start_dist = arm_length * fade_start_ratio
    fade_end_dist = arm_length * fade_end_ratio
    fade_range = fade_end_dist - fade_start_dist
    
    if fade_range < 1e-6:
        fade_range = 1e-6  
    
   
    fade_factor = 1.0 - np.clip((proj_dist - fade_start_dist) / fade_range, 0.0, 1.0)
    
    # Apply fade to alpha channel
    alpha_faded = temple_alpha_2d * fade_factor
    
    # Restore shape
    if temple_alpha.ndim == 3:
        alpha_faded = alpha_faded.reshape(h, w, 1)
    
    return alpha_faded.astype(temple_alpha.dtype)


def create_face_mask(lm, frame_shape, expand=25.0, feather=10.0):
   
    H, W = frame_shape[:2]
    
    # MediaPipe FACEMESH_FACE_OVAL Contour Index
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    # Obtain the pixel coordinates of contour points
    contour_points = []
    for idx in face_oval_indices:
        x_norm = lm[idx].x
        y_norm = lm[idx].y
        x_px = int(x_norm * W)
        y_px = int(y_norm * H)
        contour_points.append([x_px, y_px])
    
    contour_points = np.array(contour_points, dtype=np.int32)
    
    # Calculate the center of the contour
    center_x = np.mean(contour_points[:, 0])
    center_y = np.mean(contour_points[:, 1])
    center = np.array([center_x, center_y])
    
    # Extend the outline outward
    if expand > 0:
        expanded_points = []
        for pt in contour_points:
            vec = pt - center
            # Normalization
            vec_len = np.linalg.norm(vec)
            if vec_len > 0:
                vec_norm = vec / vec_len
                expanded_pt = pt + vec_norm * expand
                expanded_points.append(expanded_pt)
            else:
                expanded_points.append(pt)
        contour_points = np.array(expanded_points, dtype=np.int32)
    
    # Create a base mask
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(mask, [contour_points], 1.0)
    
    # Edge Feathering (Using Gaussian Blur)
    if feather > 0:
        # Calculate the appropriate kernel size (which must be an odd number)
        kernel_size = int(feather * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)
        
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), feather / 3.0)
    
    return mask


def verify_database_paths():
    face_ok = FACE_DB_PATH is None or os.path.exists(FACE_DB_PATH)
    eye_ok = EYE_DB_PATH is None or os.path.exists(EYE_DB_PATH)
    
    if FACE_DB_PATH and not face_ok:
        print(f"[WARNING] Face database NOT found: {FACE_DB_PATH}")
    elif FACE_DB_PATH:
        print(f"[INFO] Face database found: {FACE_DB_PATH}")
    
    if EYE_DB_PATH and not eye_ok:
        print(f"[WARNING] Eye database NOT found: {EYE_DB_PATH}")
    elif EYE_DB_PATH:
        print(f"[INFO] Eye database found: {EYE_DB_PATH}")
    
    return face_ok, eye_ok

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[WARNING] pandas not available. Using default thresholds for face/eye classification.")

class FaceShapeClassifier:
    """
    Multi-class face shape classifier (5 types)
    Classifies faces into: round, oval, oblong, square, heart
    """
    
    def __init__(self, database_path: Optional[str] = None):
        # Optimized thresholds based on actual database statistics
        self.thresholds = {
            # Length-width ratio boundaries (primary classifier)
            'round_max': 0.975,      # Round: L/W < 0.975 (very short face)
            'oval_min': 0.88,        # Oval: starts from small values
            'oval_max': 1.15,        # Oval: ends before oblong
            'oblong_min': 1.20,      # Oblong: L/W > 1.20 (clearly long face)
            
            # Secondary classifiers for middle range (0.975 < L/W < 1.20)
            'heart_square_boundary': 1.05,  # Below: likely square; Above: likely heart
            
            # Jaw-cheek ratio boundaries
            'square_jaw_min': 0.95,   # Square: jaw width ≥ 95% of cheek width
            'heart_jaw_max': 0.85,    # Heart: jaw width ≤ 85% of cheek width
            
            # Jaw-forehead ratio boundaries (for heart detection)
            'heart_jaw_forehead_max': 0.90,  # Heart: jaw much narrower than forehead
            'square_jaw_forehead_min': 0.85, # Square: jaw not too narrow vs forehead
        }
        
        if database_path and PANDAS_AVAILABLE:
            self._load_from_database(database_path)
    
    def _load_from_database(self, db_path: str):
        """Load thresholds from database statistics"""
        try:
            if not os.path.exists(db_path):
                print(f"[WARNING] Face database not found: {db_path}")
                print("[INFO] Using default thresholds")
                return
            
            print(f"[INFO] Loading face shape database (5 classes)...")
            face_db = pd.read_excel(db_path, sheet_name=None)
            
            # Process each face type
            for face_type in ['round', 'oval', 'oblong', 'square', 'heart']:
                if face_type not in face_db:
                    print(f"[WARNING] '{face_type}' sheet not found in database")
                    continue
                
                df = face_db[face_type]
                
                # Calculate statistics for this face type
                if 'length_width' in df.columns:
                    lw_values = df['length_width'].dropna()
                    lw_mean = lw_values.mean()
                    lw_p25 = lw_values.quantile(0.25)
                    lw_p75 = lw_values.quantile(0.75)
                    
                    print(f"[INFO] {face_type.upper()}: length_width = {lw_mean:.3f} [{lw_p25:.3f}, {lw_p75:.3f}]")
                
                if 'cheek_jaw' in df.columns:
                    cj_values = df['cheek_jaw'].dropna()
                    cj_mean = cj_values.mean()
                    print(f"       cheek_jaw = {cj_mean:.3f}")
                
                if 'jaw_forehead' in df.columns:
                    jf_values = df['jaw_forehead'].dropna()
                    jf_mean = jf_values.mean()
                    print(f"       jaw_forehead = {jf_mean:.3f}")
            
            # Calculate decision boundaries
            self._compute_boundaries(face_db)
            
        except Exception as e:
            print(f"[ERROR] Failed to load face database: {e}")
            print("[INFO] Using default thresholds")
    
    def _compute_boundaries(self, face_db):
        """Compute decision boundaries from actual database statistics"""
        try:
            # Statistics storage
            stats = {}
            
            # Collect statistics for each face type
            for face_type in ['round', 'oval', 'oblong', 'square', 'heart']:
                if face_type not in face_db:
                    continue
                
                df = face_db[face_type]
                stats[face_type] = {}
                
                # Length-width ratio
                if 'length_width' in df.columns:
                    lw = df['length_width'].dropna()
                    stats[face_type]['lw_mean'] = lw.mean()
                    stats[face_type]['lw_p10'] = lw.quantile(0.10)
                    stats[face_type]['lw_p25'] = lw.quantile(0.25)
                    stats[face_type]['lw_p75'] = lw.quantile(0.75)
                    stats[face_type]['lw_p90'] = lw.quantile(0.90)
                
                # Cheek-jaw ratio
                if 'cheek_jaw' in df.columns:
                    cj = df['cheek_jaw'].dropna()
                    stats[face_type]['cj_mean'] = cj.mean()
                    stats[face_type]['cj_p25'] = cj.quantile(0.25)
                    stats[face_type]['cj_p75'] = cj.quantile(0.75)
                
                # Jaw-forehead ratio
                if 'jaw_forehead' in df.columns:
                    jf = df['jaw_forehead'].dropna()
                    stats[face_type]['jf_mean'] = jf.mean()
                    stats[face_type]['jf_p25'] = jf.quantile(0.25)
                    stats[face_type]['jf_p75'] = jf.quantile(0.75)
            
            print(f"\n[INFO] Database statistics:")
            for face_type, data in stats.items():
                print(f"  {face_type.upper()}:")
                for key, val in data.items():
                    print(f"    {key}: {val:.3f}")
            
            # Compute decision boundaries
            
            # 1. Round threshold (upper bound)
            if 'round' in stats and 'lw_p90' in stats['round']:
                self.thresholds['round_max'] = min(0.98, stats['round']['lw_p90'])
            
            # 2. Oblong threshold (lower bound)
            if 'oblong' in stats and 'lw_p10' in stats['oblong']:
                self.thresholds['oblong_min'] = max(1.15, stats['oblong']['lw_p10'])
            
            # 3. Oval range
            if 'oval' in stats:
                if 'lw_p10' in stats['oval']:
                    self.thresholds['oval_min'] = stats['oval']['lw_p10']
                if 'lw_p90' in stats['oval']:
                    self.thresholds['oval_max'] = stats['oval']['lw_p90']
            
            # 4. Square jaw threshold
            if 'square' in stats and 'cj_p25' in stats['square']:
                self.thresholds['square_jaw_min'] = max(0.92, stats['square']['cj_p25'])
            
            # 5. Heart jaw threshold (narrow jaw)
            if 'heart' in stats and 'cj_p75' in stats['heart']:
                self.thresholds['heart_jaw_max'] = min(0.88, stats['heart']['cj_p75'])
            
            # 6. Heart forehead threshold (based on jaw-forehead ratio)
            if 'heart' in stats and 'jf_p75' in stats['heart']:
                # Heart has narrow jaw relative to forehead (low jf ratio)
                self.thresholds['heart_jaw_forehead_max'] = min(0.92, stats['heart']['jf_p75'])
            
            # 7. Square forehead threshold
            if 'square' in stats and 'jf_p25' in stats['square']:
                self.thresholds['square_jaw_forehead_min'] = max(0.80, stats['square']['jf_p25'])
            
            print(f"\n[INFO] Optimized decision boundaries:")
            for key, val in sorted(self.thresholds.items()):
                print(f"  {key}: {val:.3f}")
            
        except Exception as e:
            print(f"[WARNING] Error computing boundaries: {e}")
            import traceback
            traceback.print_exc()
    
    def classify(self, landmarks, W, H) -> Tuple[str, float, Dict]:
        """
        Classify face shape into 5 categories using decision tree
        Priority: Oblong > Round > (Heart vs Square in middle) > Oval
        Returns: (face_type, confidence, features_dict)
        """
        # Extract face measurements
        forehead_left = landmarks[21]
        forehead_right = landmarks[251]
        forehead_width = abs(forehead_right.x - forehead_left.x) * W
        
        cheek_left = landmarks[234]
        cheek_right = landmarks[454]
        cheek_width = abs(cheek_right.x - cheek_left.x) * W
        
        jaw_left = landmarks[172]
        jaw_right = landmarks[397]
        jaw_width = abs(jaw_right.x - jaw_left.x) * W
        
        forehead_center = landmarks[10]
        chin = landmarks[152]
        face_height = abs(chin.y - forehead_center.y) * H
        
        # Calculate ratios
        face_width = cheek_width
        length_width_ratio = face_height / (face_width + 1e-6)
        cheek_jaw_ratio = jaw_width / (cheek_width + 1e-6)
        jaw_forehead_ratio = jaw_width / (forehead_width + 1e-6)
        
        features = {
            'face_width': face_width,
            'face_height': face_height,
            'length_width': length_width_ratio,
            'cheek_width': cheek_width,
            'forehead_width': forehead_width,
            'jaw_width': jaw_width,
            'cheek_jaw': cheek_jaw_ratio,
            'jaw_forehead': jaw_forehead_ratio,
        }
        
        # ===== DECISION TREE =====
        
        # 1. Check for OBLONG (L/W > 1.20)
        if length_width_ratio >= self.thresholds['oblong_min']:
            # Very long face
            distance_above = length_width_ratio - self.thresholds['oblong_min']
            confidence = min(0.95, 0.70 + distance_above / 0.3)
            return 'oblong', confidence, features
        
        # 2. Check for ROUND (L/W < 0.975)
        if length_width_ratio <= self.thresholds['round_max']:
            # Very short face
            distance_below = self.thresholds['round_max'] - length_width_ratio
            confidence = min(0.95, 0.70 + distance_below / 0.15)
            return 'round', confidence, features
        
        # 3. MIDDLE RANGE (0.975 < L/W < 1.20)
        # Need to distinguish: Oval, Square, Heart
        
        # 3a. Check for HEART shape
        # Heart characteristics: narrow jaw + wide forehead
        has_narrow_jaw_vs_cheek = cheek_jaw_ratio <= self.thresholds['heart_jaw_max']
        has_narrow_jaw_vs_forehead = jaw_forehead_ratio <= self.thresholds['heart_jaw_forehead_max']
        
        if has_narrow_jaw_vs_cheek and has_narrow_jaw_vs_forehead:
            # Strong heart indicators
            jaw_narrowness = (self.thresholds['heart_jaw_max'] - cheek_jaw_ratio) / 0.15
            forehead_prominence = (self.thresholds['heart_jaw_forehead_max'] - jaw_forehead_ratio) / 0.15
            confidence = min(0.90, 0.65 + (jaw_narrowness + forehead_prominence) / 2 * 0.25)
            return 'heart', confidence, features
        
        elif has_narrow_jaw_vs_cheek or has_narrow_jaw_vs_forehead:
            # Moderate heart indicators
            return 'heart', 0.60, features
        
        # 3b. Check for SQUARE shape
        # Square characteristics: wide jaw + balanced proportions
        has_wide_jaw = cheek_jaw_ratio >= self.thresholds['square_jaw_min']
        has_balanced_jaw_forehead = jaw_forehead_ratio >= self.thresholds['square_jaw_forehead_min']
        
        if has_wide_jaw and has_balanced_jaw_forehead:
            # Strong square indicators
            jaw_width_factor = (cheek_jaw_ratio - 0.90) / 0.15
            confidence = min(0.90, 0.65 + jaw_width_factor * 0.25)
            return 'square', confidence, features
        
        elif has_wide_jaw:
            # Moderate square indicator
            return 'square', 0.60, features
        
        # 3c. OVAL (fallback for middle range)
        # Check if in core oval range
        in_oval_range = (
            self.thresholds['oval_min'] <= length_width_ratio <= self.thresholds['oval_max']
        )
        
        if in_oval_range:
            # Calculate centrality in oval range
            oval_center = (self.thresholds['oval_min'] + self.thresholds['oval_max']) / 2
            distance_from_center = abs(length_width_ratio - oval_center)
            oval_span = (self.thresholds['oval_max'] - self.thresholds['oval_min']) / 2
            
            centrality = 1.0 - (distance_from_center / oval_span)
            confidence = max(0.55, min(0.85, 0.60 + centrality * 0.25))
            return 'oval', confidence, features
        
        # Edge case: between ranges
        if length_width_ratio > self.thresholds['oval_max']:
            # Closer to oblong
            return 'oval', 0.55, features
        else:
            # Closer to round
            return 'oval', 0.55, features


class FrameFaceAdapter:
    """
    Frame adaptation rules for 5 face shapes
    """
    
    def __init__(self):
        self.face_shape_rules = {
            'round': {
                'scale_multiplier': 1.05,
                'width_adjust': 1.08,
                'height_adjust': 0.95,
                'vertical_offset': 2.0,
                'temple_offset_factor': 1.1,
            },
            'oval': {
                'scale_multiplier': 0.98,
                'width_adjust': 0.95,
                'height_adjust': 1.02,
                'vertical_offset': -3.0,
                'temple_offset_factor': 0.98,
            },
            'oblong': {
                'scale_multiplier': 1.02,
                'width_adjust': 1.05,
                'height_adjust': 0.92,  # Reduce height to balance long face
                'vertical_offset': 0.0,
                'temple_offset_factor': 1.05,
            },
            'square': {
                'scale_multiplier': 1.00,
                'width_adjust': 1.02,
                'height_adjust': 1.00,
                'vertical_offset': -1.0,
                'temple_offset_factor': 1.00,
            },
            'heart': {
                'scale_multiplier': 0.96,
                'width_adjust': 0.98,   # Slightly narrower to match narrow jaw
                'height_adjust': 1.03,
                'vertical_offset': -2.0,
                'temple_offset_factor': 0.95,
            }
        }
        
        # Eye shape rules remain the same
        self.eye_shape_rules = {
            'almond_upturned': {
                'frame_tilt': -2.0,
                'temple_tilt': -3.0,
                'vertical_nudge': -2.0,
            },
            'almond_downturned': {
                'frame_tilt': 1.5,
                'temple_tilt': 2.0,
                'vertical_nudge': 1.0,
            },
            'almond_neutral': {
                'frame_tilt': 0.0,
                'temple_tilt': 0.0,
                'vertical_nudge': 0.0,
            },
            'round_open': {
                'frame_tilt': 0.0,
                'temple_tilt': 0.0,
                'vertical_nudge': -3.0,
            }
        }
        
    def compute_frame_params(self, 
                            face_shape: str, 
                            eye_shape: str,
                            base_ipd: float,
                            frame_id: Optional[str] = None) -> Dict:
        """
        Compute frame adaptation parameters for given face and eye shape
        """
        face_rules = self.face_shape_rules.get(face_shape, self.face_shape_rules['oval'])
        eye_rules = self.eye_shape_rules.get(eye_shape, self.eye_shape_rules['almond_neutral'])
        
        params = {
            # Dimension Parameters
            'scale_multiplier': face_rules['scale_multiplier'],
            'width_factor': face_rules['width_adjust'],
            'height_factor': face_rules['height_adjust'],
            
            # Position parameters
            'vertical_offset': face_rules['vertical_offset'] + eye_rules['vertical_nudge'],
            'horizontal_offset': 0.0,
            
            # Angular parameter
            'frame_tilt': eye_rules['frame_tilt'],
            
            # Temple Parameters
            'temple_rotation': eye_rules['temple_tilt'],
            'temple_offset_left': face_rules['temple_offset_factor'],
            'temple_offset_right': face_rules['temple_offset_factor'],
            
            # Metadata
            'face_shape': face_shape,
            'eye_shape': eye_shape,
            'base_ipd': base_ipd,
        }
        
        return params


class EyeShapeClassifier:
    
    def __init__(self, database_path: Optional[str] = None):
     
        self.almond_aspect_ratio = (4.5, 7.0)
        self.round_aspect_ratio = (3.0, 5.0)
        
        if database_path and PANDAS_AVAILABLE:
            self._load_from_database(database_path)
    
    def _load_from_database(self, db_path: str):
        try:
            if not os.path.exists(db_path):
                print(f"[WARNING] Eye database not found: {db_path}")
                print("[INFO] Using default thresholds")
                return
            
            print(f"[INFO] Loading eye shape database...")
            eye_db = pd.read_excel(db_path, sheet_name=None)
            
            almond_ratios = []
            round_ratios = []
            
            for sheet_name, df in eye_db.items():
                if 'aspect_ratio' in df.columns:
                    ratios = df['aspect_ratio'].dropna()
                    
                    if 'almond' in sheet_name.lower():
                        almond_ratios.extend(ratios.tolist())
                    elif 'round' in sheet_name.lower():
                        round_ratios.extend(ratios.tolist())
            
            if almond_ratios:
                almond_arr = np.array(almond_ratios)
                almond_min = np.percentile(almond_arr, 5)  
                almond_max = np.percentile(almond_arr, 95)  
                self.almond_aspect_ratio = (almond_min, almond_max)
                print(f"[INFO] Almond eye aspect ratio: ({almond_min:.2f}, {almond_max:.2f}) [5th-95th percentile]")
            
            if round_ratios:
                round_arr = np.array(round_ratios)
                round_min = np.percentile(round_arr, 5)  
                round_max = np.percentile(round_arr, 95)  
                self.round_aspect_ratio = (round_min, round_max)
                print(f"[INFO] Round eye aspect ratio: ({round_min:.2f}, {round_max:.2f}) [5th-95th percentile]")
                
        except Exception as e:
            print(f"[ERROR] Failed to load eye database: {e}")
            print("[INFO] Using default thresholds")
        
    def classify(self, landmarks, W, H) -> Tuple[str, float, Dict]:
   
        # Left Eye Key Point
        left_outer = landmarks[33]
        left_inner = landmarks[133]
        left_top = landmarks[159]
        left_bottom = landmarks[145]
        
        # Right Eye Key Point
        right_outer = landmarks[263]
        right_inner = landmarks[362]
        right_top = landmarks[386]
        right_bottom = landmarks[374]
        
        # Calculate the width and height of the left eye
        left_width = abs(left_outer.x - left_inner.x) * W
        left_height = abs(left_top.y - left_bottom.y) * H
        
        # Calculate the width and height of the right eye
        right_width = abs(right_outer.x - right_inner.x) * W
        right_height = abs(right_top.y - right_bottom.y) * H
        
        # The database uses height/width, so the reciprocal is used here.
        left_aspect_inv = left_height / (left_width + 1e-6)
        right_aspect_inv = right_height / (right_width + 1e-6)
        avg_aspect = (left_aspect_inv + right_aspect_inv) / 2.0
        
        # Calculate the eye tilt angle
        left_tilt = math.degrees(math.atan2(
            (left_outer.y - left_inner.y) * H,
            (left_outer.x - left_inner.x) * W
        ))
        right_tilt = math.degrees(math.atan2(
            (right_inner.y - right_outer.y) * H,
            (right_inner.x - right_outer.x) * W
        ))
        avg_tilt = (left_tilt + right_tilt) / 2.0
        
        features = {
            'left_width': left_width,
            'left_height': left_height,
            'right_width': right_width,
            'right_height': right_height,
            'aspect_ratio': avg_aspect,  # height/width
            'tilt_deg': avg_tilt,
        }
        
        import sys
        if not hasattr(sys, '_eye_classify_count'):
            sys._eye_classify_count = 0
        
        if sys._eye_classify_count < 3:
            print(f"\n[EYE DEBUG #{sys._eye_classify_count}]")
            print(f"  Computed aspect: {avg_aspect:.4f} (height/width - INVERTED)")
            print(f"  DB almond range: {self.almond_aspect_ratio}")
            print(f"  DB round range: {self.round_aspect_ratio}")
            print(f"  Left: h={left_height:.1f}/w={left_width:.1f} = {left_aspect_inv:.4f}")
            print(f"  Right: h={right_height:.1f}/w={right_width:.1f} = {right_aspect_inv:.4f}")
            print(f"  [Note: using inverse to match database definition]")
            sys._eye_classify_count += 1
        
        almond_min, almond_max = self.almond_aspect_ratio
        margin = (almond_max - almond_min) * 0.05  
        
        if almond_min <= avg_aspect <= almond_max:
            if avg_tilt > 2.0:
                result = ('almond_upturned', 0.8, features)
            elif avg_tilt < -2.0:
                result = ('almond_downturned', 0.8, features)
            else:
                result = ('almond_neutral', 0.8, features)
            if sys._eye_classify_count <= 3:
                print(f"  → IN almond range: {result[0]} (conf={result[1]})")
            return result
        
        # Near the border
        elif almond_min - margin <= avg_aspect <= almond_max + margin:
            if avg_tilt > 2.0:
                result = ('almond_upturned', 0.7, features)
            elif avg_tilt < -2.0:
                result = ('almond_downturned', 0.7, features)
            else:
                result = ('almond_neutral', 0.7, features)
            if sys._eye_classify_count <= 3:
                print(f"  → NEAR almond range: {result[0]} (conf={result[1]}) [boundary case]")
            return result
        
        # Significantly below the range
        elif avg_aspect < almond_min:
            if sys._eye_classify_count <= 3:
                print(f"  → BELOW almond range: round_open")
            return 'round_open', 0.7, features
        
        # Significantly above the range
        else:
            if sys._eye_classify_count <= 3:
                print(f"  → FAR ABOVE almond range: almond_neutral (fallback)")
            return 'almond_neutral', 0.5, features


class FrameFaceAdapter:
    
    def __init__(self):
        self.face_shape_rules = {
            'round': {
                'scale_multiplier': 1.05,
                'width_adjust': 1.08,
                'height_adjust': 0.95,
                'vertical_offset': 2.0,
                'temple_offset_factor': 1.1,
            },
            'oval': {
                'scale_multiplier': 0.98,
                'width_adjust': 0.95,
                'height_adjust': 1.02,
                'vertical_offset': -3.0,
                'temple_offset_factor': 0.98,
            }
        }
        
        self.eye_shape_rules = {
            'almond_upturned': {
                'frame_tilt': -2.0,
                'temple_tilt': -3.0,
                'vertical_nudge': -2.0,
            },
            'almond_downturned': {
                'frame_tilt': 1.5,
                'temple_tilt': 2.0,
                'vertical_nudge': 1.0,
            },
            'almond_neutral': {
                'frame_tilt': 0.0,
                'temple_tilt': 0.0,
                'vertical_nudge': 0.0,
            },
            'round_open': {
                'frame_tilt': 0.0,
                'temple_tilt': 0.0,
                'vertical_nudge': -3.0,
            }
        }
        
    def compute_frame_params(self, 
                            face_shape: str, 
                            eye_shape: str,
                            base_ipd: float,
                            frame_id: Optional[str] = None) -> Dict:

        face_rules = self.face_shape_rules.get(face_shape, self.face_shape_rules['oval'])
        eye_rules = self.eye_shape_rules.get(eye_shape, self.eye_shape_rules['almond_neutral'])
        
        params = {
            # Dimension Parameters
            'scale_multiplier': face_rules['scale_multiplier'],
            'width_factor': face_rules['width_adjust'],
            'height_factor': face_rules['height_adjust'],
            
            # Position parameters
            'vertical_offset': face_rules['vertical_offset'] + eye_rules['vertical_nudge'],
            'horizontal_offset': 0.0,
            
            # Angular parameter
            'frame_tilt': eye_rules['frame_tilt'],
            
            # Temple Parameters
            'temple_rotation': eye_rules['temple_tilt'],
            'temple_offset_left': face_rules['temple_offset_factor'],
            'temple_offset_right': face_rules['temple_offset_factor'],
            
            # Metadata
            'face_shape': face_shape,
            'eye_shape': eye_shape,
            'base_ipd': base_ipd,
        }
        
        return params


class SmartFrameFaceMatcher:
    
    def __init__(self, 
                 face_db_path: Optional[str] = None,
                 eye_db_path: Optional[str] = None):

        print("[INFO] Initializing SmartFrameFaceMatcher...")
        
        self.face_classifier = FaceShapeClassifier(database_path=face_db_path)
        self.eye_classifier = EyeShapeClassifier(database_path=eye_db_path)
        self.adapter = FrameFaceAdapter()
        
        # Classification Lock
        self.classification_locked = False
        
        # Cache identification results
        self.face_shape_cache = None
        self.eye_shape_cache = None
        self.confidence_threshold = 0.5  
        self.update_counter = 0
        self.update_interval = 30  
        
        print("[INFO] SmartFrameFaceMatcher initialized!")
        
    def analyze_and_fit(self, landmarks, W, H, current_ipd: float) -> Dict:
       
        # After locking, skip classification and cache directly
        should_classify = not self.classification_locked

        if should_classify and (self.update_counter == 0 or self.update_counter % self.update_interval == 0):
            print(f"\n{'='*60}")
            print(f"RUNNING CLASSIFICATION (frame {self.update_counter})")
            print(f"{'='*60}")
            
            face_shape, face_conf, face_features = self.face_classifier.classify(landmarks, W, H)
            eye_shape, eye_conf, eye_features = self.eye_classifier.classify(landmarks, W, H)
            
            print(f"\n[RESULTS]")
            print(f"  Face: {face_shape} (confidence={face_conf:.3f}, threshold={self.confidence_threshold})")
            print(f"  Eye: {eye_shape} (confidence={eye_conf:.3f}, threshold={self.confidence_threshold})")
            
            if face_conf >= self.confidence_threshold:
                self.face_shape_cache = (face_shape, face_conf, face_features)
                print("  ✓ Face shape CACHED")
            else:
                print("  ✗ Face confidence too low")
            
            if eye_conf >= self.confidence_threshold:
                self.eye_shape_cache = (eye_shape, eye_conf, eye_features)
                print("  ✓ Eye shape CACHED")
            else:
                print("  ✗ Eye confidence too low")
            
            # Face & Eye successfully cached and locked(subsequent entries will not be classified)
            if (
                self.face_shape_cache is not None and
                self.eye_shape_cache is not None and
                not self.classification_locked
            ):
                self.classification_locked = True
                print("[INFO] SmartFrameFaceMatcher: classification LOCKED after first successful cache.")
            
            print(f"{'='*60}\n")
        
        self.update_counter += 1
        
        # Use cached classification results，default without cache
        face_shape = self.face_shape_cache[0] if self.face_shape_cache else 'oval'
        eye_shape = self.eye_shape_cache[0] if self.eye_shape_cache else 'almond_neutral'
        
        # Calculate Adaptation Parameters
        params = self.adapter.compute_frame_params(
            face_shape=face_shape,
            eye_shape=eye_shape,
            base_ipd=current_ipd
        )
        
        params['_debug'] = {
            'face_shape': face_shape,
            'face_confidence': self.face_shape_cache[1] if self.face_shape_cache else 0.0,
            'eye_shape': eye_shape,
            'eye_confidence': self.eye_shape_cache[1] if self.eye_shape_cache else 0.0,
            'locked': self.classification_locked,
        }
        
        return params
    
    def reset(self):
        self.face_shape_cache = None
        self.eye_shape_cache = None
        self.update_counter = 0
        self.classification_locked = False

HERE = os.path.dirname(__file__)
FRAME_DIR = os.path.join(HERE, "frames")
FRAME_IDS = ("A", "B", "C", "D")
DEFAULT_FRAME_ID = "A"
SIZES = ("S", "M", "L")
DEFAULT_SIZE = "L"

def env_bool(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1","true","yes","on","y")

W_CAP = int(os.environ.get("CAP_W", "640"))
H_CAP = int(os.environ.get("CAP_H", "360"))
CAM_INDEX = int(os.environ.get("CAM_INDEX", "1"))
H_MIRROR = env_bool("H_MIRROR", "1")

SIZE_SCALE = {
    "S": float(os.environ.get("SIZE_K_S", "1.00")),
    "M": float(os.environ.get("SIZE_K_M", "1.10")),
    "L": float(os.environ.get("SIZE_K_L", "1.20")),
}

T_SM = float(os.environ.get("AUTO_FIT_T_SM", "0.24"))
T_ML = float(os.environ.get("AUTO_FIT_T_ML", "0.29"))
CENTER_T_BIAS = float(os.environ.get("CENTER_T_BIAS", "0.00"))
CENTER_N_BIAS = float(os.environ.get("CENTER_N_BIAS", "-5.0"))  

SHEAR_K = float(os.environ.get("SHEAR_K", "0.12"))
USE_Z_SHEAR = bool(int(os.environ.get("USE_Z_SHEAR", "1")))
Z_SHEAR_K = float(os.environ.get("Z_SHEAR_K", "0.50"))
SMOOTH_A = float(os.environ.get("SMOOTH_A", "0.70"))
SHOW_HUD = env_bool("SHOW_HUD", "1")

LANDMARK_MODE = os.environ.get("LANDMARK_MODE", "stable")
MANUAL_OFFSET_X = float(os.environ.get("OFFSET_X", "0.0"))
MANUAL_OFFSET_Y = float(os.environ.get("OFFSET_Y", "0.0"))
MANUAL_SCALE_ADJUST = float(os.environ.get("SCALE_ADJUST", "1.25"))
GLASSES_WIDTH_FACTOR = float(os.environ.get("GLASSES_WIDTH", "1.9"))
FRAME_HEIGHT_SCALE = float(os.environ.get("FRAME_HEIGHT_SCALE", "0.80"))
FRAME_VERTICAL_OFFSET = float(os.environ.get("FRAME_VERTICAL_OFFSET", "5.0"))
PERSPECTIVE_SCALE = float(os.environ.get("PERSP_SCALE", "0.20"))

# Interpupillary distance stabilization parameters
IPD_STABILIZATION_ENABLED = env_bool("IPD_STABILIZATION", "1")
IPD_YAW_THRESHOLD = float(os.environ.get("IPD_YAW_THRESHOLD", "40.0"))
IPD_YAW_MAX = float(os.environ.get("IPD_YAW_MAX", "60.0"))
IPD_MIN_RATIO = float(os.environ.get("IPD_MIN_RATIO", "0.85"))
IPD_SMOOTH_WINDOW = int(os.environ.get("IPD_SMOOTH_WINDOW", "30"))

ARM_SWAP = env_bool("ARM_SWAP", "1")  
ARM_FLIP_X_L = env_bool("ARM_FLIP_X_L", "0")
ARM_FLIP_X_R = env_bool("ARM_FLIP_X_R", "0")

FRAME_ARM_ADJUSTMENTS = {
    "A": (-35.0, -17.0, 20.0, -22.0),
    "B": (0.0, 0.0, -9.0, 2.5),
    "C": (-7.0, 5.0, 3.0, 10.0),
    "D": (0.0, 0.0, 0.0, 0.0),
}


FRAME_SCALE_ADJUSTMENTS = {
    "A": 1.00,  
    "B": 1.00, 
    "C": 1.00,  
    "D": 1.00,  
}

FRAME_VERTICAL_ADJUSTMENTS = {
    "A": 0.0,   
    "B": 0.0,   
    "C": -5.0,  
    "D": -3.0, 
}

ARM_SCALE_MULTIPLIER = float(os.environ.get("ARM_SCALE_MULT", "0.45"))  
ARM_OUT_PCT = float(os.environ.get("ARM_OUT_PCT", "0.08"))
ARM_OUT_MIN = float(os.environ.get("ARM_OUT_MIN", "25.0"))
ARM_OUT_MAX = float(os.environ.get("ARM_OUT_MAX", "80.0"))
HINGE_OUT_U_PX = float(os.environ.get("HINGE_OUT_U_PX", "15.0"))
HINGE_OUT_U_PCT = float(os.environ.get("HINGE_OUT_U_PCT", "0.05"))
HINGE_OUT_N_PX = float(os.environ.get("HINGE_OUT_N_PX", "5.0"))
PIVOT_Y_LOW = float(os.environ.get("PIVOT_Y_LOW", "0.35"))
PIVOT_Y_HIGH = float(os.environ.get("PIVOT_Y_HIGH", "0.65"))
ARM_LEN_K = float(os.environ.get("ARM_LEN_K", "32.0"))
ARM_LEN_MAX = float(os.environ.get("ARM_LEN_MAX", "0.30"))

PNG_TILT_COMPENSATION = float(os.environ.get("PNG_TILT_COMP", "0.0"))  
TEMPLE_ANGLE_OFFSET = float(os.environ.get("TEMPLE_ANGLE_OFFSET", "0.0"))  

# Independent angular deviation of left and right eyeglass temples
TEMPLE_ANGLE_OFFSET_L = float(os.environ.get("TEMPLE_ANGLE_OFFSET_L", "180.0")) 
TEMPLE_ANGLE_OFFSET_R = float(os.environ.get("TEMPLE_ANGLE_OFFSET_R", "180.0")) 

# Temple width adjustment
TEMPLE_WIDTH_SCALE = float(os.environ.get("TEMPLE_WIDTH_SCALE", "0.3"))  
TEMPLE_WIDTH_SCALE_L = float(os.environ.get("TEMPLE_WIDTH_SCALE_L", str(TEMPLE_WIDTH_SCALE)))  
TEMPLE_WIDTH_SCALE_R = float(os.environ.get("TEMPLE_WIDTH_SCALE_R", str(TEMPLE_WIDTH_SCALE)))  

# Fade out the part of the eyeglass arm that extends beyond the anchor point
TEMPLE_FADEOUT_ENABLED = bool(int(os.environ.get("TEMPLE_FADEOUT", "1")))  
TEMPLE_FADEOUT_START = float(os.environ.get("TEMPLE_FADEOUT_START", "0.85"))  
TEMPLE_FADEOUT_END = float(os.environ.get("TEMPLE_FADEOUT_END", "1.15"))  


# When facing forward, use a more aggressive fade to let the portion extending beyond the ears naturally disappear
ADAPTIVE_FADEOUT_ENABLED = env_bool("ADAPTIVE_FADEOUT", "1")  
FRONT_FADEOUT_START_RATIO = float(os.environ.get("FRONT_FADEOUT_START_RATIO", "0.65"))  
FRONT_FADEOUT_END_RATIO = float(os.environ.get("FRONT_FADEOUT_END_RATIO", "1.00"))      
SIDE_FADEOUT_START_RATIO = float(os.environ.get("SIDE_FADEOUT_START_RATIO", "0.85"))    
SIDE_FADEOUT_END_RATIO = float(os.environ.get("SIDE_FADEOUT_END_RATIO", "1.15"))       
ADAPTIVE_YAW_START = float(os.environ.get("ADAPTIVE_YAW_START", "15.0"))  
ADAPTIVE_YAW_END = float(os.environ.get("ADAPTIVE_YAW_END", "30.0"))      

LEFT_INWARD_ADJUST = float(os.environ.get("LEFT_IN_ADJ", "27.0"))
LEFT_UPWARD_ADJUST = float(os.environ.get("LEFT_UP_ADJ", "-48.0"))
RIGHT_OUTWARD_ADJUST = float(os.environ.get("RIGHT_OUT_ADJ", "35.0"))
RIGHT_UPWARD_ADJUST = float(os.environ.get("RIGHT_UP_ADJ", "-48.0"))

INWARD_BASE = float(os.environ.get("INWARD_BASE", "16.0"))
PERSPECTIVE_FACTOR = float(os.environ.get("PERSP_FACTOR", "0.18"))
BACKWARD_BASE = float(os.environ.get("BACKWARD_BASE", "12.0"))
RIGHT_OUTWARD = float(os.environ.get("RIGHT_OUTWARD", "0.0"))
RIGHT_UPWARD = float(os.environ.get("RIGHT_UPWARD", "12.0"))

EXTRA_OFFSET_U_L = float(os.environ.get("EXTRA_U_L", "6.0"))
EXTRA_OFFSET_N_L = float(os.environ.get("EXTRA_N_L", "-5.0"))
EXTRA_OFFSET_U_R = float(os.environ.get("EXTRA_U_R", "-5.0"))
EXTRA_OFFSET_N_R = float(os.environ.get("EXTRA_N_R", "-3.0"))

OCCLUSION_ENABLED = env_bool("OCCLUSION_ENABLED", "1")
FRONT_HIDE_THRESHOLD = float(os.environ.get("FRONT_HIDE_THRESHOLD", "12.0"))
FRONT_SHOW_THRESHOLD = float(os.environ.get("FRONT_SHOW_THRESHOLD", "20.0"))
SIDE_HIDE_START = float(os.environ.get("SIDE_HIDE_START", "15.0"))
SIDE_HIDE_COMPLETE = float(os.environ.get("SIDE_HIDE_COMPLETE", "35.0"))



FRONT_TEMPLE_FADEOUT_ENABLED = env_bool("FRONT_TEMPLE_FADEOUT", "0")
FRONT_FADEOUT_START = float(os.environ.get("FRONT_FADEOUT_START", "15.0")) 
FRONT_FADEOUT_END = float(os.environ.get("FRONT_FADEOUT_END", "30.0"))     

# Face contour mask (masking the mirror arm sections extending beyond the facial area)
FACE_MASK_ENABLED = env_bool("FACE_MASK_ENABLED", "1")  
FACE_MASK_EXPAND = float(os.environ.get("FACE_MASK_EXPAND", "8.0"))  
FACE_MASK_FEATHER = float(os.environ.get("FACE_MASK_FEATHER", "5.0"))  

mp_fm = mp.solutions.face_mesh
fm = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=False, 
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

def to_img_px(lm, W, H):
    return np.array([lm.x * W, lm.y * H], np.float32)

def clamp(x, a, b): 
    return a if x < a else (b if x > b else x)

def ema(prev, new, a): 
    return new if prev is None else (a*prev + (1.0-a)*new)

def smooth_angle(prev, cur, a):
    if prev is None: return cur
    delta = (cur - prev + math.pi) % (2*math.pi) - math.pi
    return prev + (1.0 - a) * delta

def estimate_head_pose(landmarks, W, H):
    nose_tip = landmarks[1]
    chin = landmarks[152]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    
    cheek_dx = (right_cheek.x - left_cheek.x) * W
    cheek_dz = right_cheek.z - left_cheek.z
    yaw = math.atan2(cheek_dz, cheek_dx / W * 0.5)
    
    pitch_dy = (chin.y - nose_tip.y) * H
    pitch_dz = chin.z - nose_tip.z
    pitch = math.atan2(pitch_dz, pitch_dy / H * 0.3)
    
    left_eye = landmarks[263]
    right_eye = landmarks[33]
    eye_dy = (right_eye.y - left_eye.y) * H
    eye_dx = (right_eye.x - left_eye.x) * W
    roll = math.atan2(eye_dy, eye_dx)
    
    return yaw, pitch, roll

def get_z_for_point(landmarks, indices):
    return sum(landmarks[idx].z for idx in indices) / len(indices)

class IPDStabilizer:
    def __init__(self, window_size=30, yaw_threshold=40.0, yaw_max=60.0, min_ratio=0.85):
        self.window_size = window_size
        self.yaw_threshold = yaw_threshold
        self.yaw_max = yaw_max
        self.min_ratio = min_ratio
        
        self.ipd_history = []
        self.baseline_ipd = None
        self.baseline_d = None
        
    def update(self, raw_ipd, yaw_rad, glasses_width_factor=1.9):
        yaw_deg = abs(math.degrees(yaw_rad))
        
        # Stabilization disabled at extreme angles
        EXTREME_SIDE_THRESHOLD = 50.0
        if yaw_deg > EXTREME_SIDE_THRESHOLD:
            return raw_ipd
        
        if yaw_deg < self.yaw_threshold:
            self.ipd_history.append(raw_ipd)
            if len(self.ipd_history) > self.window_size:
                self.ipd_history.pop(0)
            
            if len(self.ipd_history) >= 5:
                self.baseline_ipd = np.median(self.ipd_history)
                self.baseline_d = self.baseline_ipd * glasses_width_factor
        
        if self.baseline_ipd is None:
            return raw_ipd
        
        if yaw_deg <= self.yaw_threshold:
            return raw_ipd
        
        # Compensation Calculation
        if yaw_deg >= self.yaw_max:
            compensation_ratio = 1.0
        else:
            compensation_ratio = (yaw_deg - self.yaw_threshold) / (self.yaw_max - self.yaw_threshold)
        
        min_allowed_ipd = self.baseline_ipd * self.min_ratio
        
        if raw_ipd >= min_allowed_ipd:
            return raw_ipd
        
        stabilized_ipd = raw_ipd + (min_allowed_ipd - raw_ipd) * compensation_ratio
        return stabilized_ipd
    
    def reset(self):
        self.ipd_history.clear()
        self.baseline_ipd = None
        self.baseline_d = None

def get_glasses_anchor_points_stable(landmarks, W, H, yaw_rad=0.0, ipd_stabilizer=None):
    nose_bridge = landmarks[168]
    left_inner_eye = landmarks[133]
    right_inner_eye = landmarks[362]
    left_outer_eye = landmarks[33]
    right_outer_eye = landmarks[263]
    
    nose_px = to_img_px(nose_bridge, W, H)
    left_inner_px = to_img_px(left_inner_eye, W, H)
    right_inner_px = to_img_px(right_inner_eye, W, H)
    left_outer_px = to_img_px(left_outer_eye, W, H)
    right_outer_px = to_img_px(right_outer_eye, W, H)
    
    inner_center = 0.5 * (left_inner_px + right_inner_px)
    
    # Calculate the original IPD
    raw_inner_eye_distance = float(np.linalg.norm(right_inner_px - left_inner_px))
    
    # Use the IPD stabilizer
    if IPD_STABILIZATION_ENABLED and ipd_stabilizer is not None:
        stabilized_ipd = ipd_stabilizer.update(raw_inner_eye_distance, yaw_rad, GLASSES_WIDTH_FACTOR)
    else:
        stabilized_ipd = raw_inner_eye_distance
    
    inner_eye_distance = stabilized_ipd
    ipd = stabilized_ipd
    
    glasses_width_factor = GLASSES_WIDTH_FACTOR
    glasses_half_width = inner_eye_distance * glasses_width_factor / 2.0
    
    inner_vec = right_inner_px - left_inner_px
    inner_vec_norm = inner_vec / (np.linalg.norm(inner_vec) + 1e-6)
    
    perp_vec = np.array([-inner_vec_norm[1], inner_vec_norm[0]], np.float32)
    
    vertical_offset = inner_eye_distance * 0.05
    horizontal_offset = inner_eye_distance * 0.15
    
    base_center = 0.6 * nose_px + 0.4 * inner_center
    
    left_anchor = (base_center 
                   - inner_vec_norm * (glasses_half_width + horizontal_offset)
                   + perp_vec * vertical_offset)
    
    right_anchor = (base_center 
                    + inner_vec_norm * (glasses_half_width + horizontal_offset)
                    + perp_vec * vertical_offset)
    
    left_indices = [33, 133]
    right_indices = [263, 362]
    
    return left_anchor, right_anchor, left_indices, right_indices, ipd


def _read_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        a = np.full(img.shape[:2], 255, np.uint8)
        img = np.dstack([img, a])
    return img

def _alpha_band_edges(mask, y0p=0.35, y1p=0.65):
    h, w = mask.shape
    y0 = max(0, min(h-1, int(round(y0p*h))))
    y1 = max(0, min(h-1, int(round(y1p*h))))
    if y1 <= y0: y0, y1 = 0, h-1
    xsL, xsR = [], []
    for y in range(y0, y1+1):
        nz = np.flatnonzero(mask[y])
        if nz.size:
            xsL.append(nz[0]); xsR.append(nz[-1])
    if not xsL:
        return (int(0.2*w), int((y0+y1)/2)), (int(0.8*w), int((y0+y1)/2))
    return (int(np.median(xsL)), int((y0+y1)/2)), (int(np.median(xsR)), int((y0+y1)/2))

def _get_outermost_edges(mask):
    h, w = mask.shape
    
    row_data = []
    for y in range(h):
        nz = np.flatnonzero(mask[y])
        if nz.size > 0:
            left_x = nz[0]
            right_x = nz[-1]
            width = right_x - left_x
            row_data.append((y, left_x, right_x, width))
    
    if not row_data:
        return (int(0.2*w), h//2), (int(0.8*w), h//2)
    
    widths = [item[3] for item in row_data]
    max_width_idx = np.argmax(widths)
    y_widest, left_widest, right_widest, _ = row_data[max_width_idx]
    
    y_start = max(0, int(y_widest - 0.2 * h))
    y_end = min(h, int(y_widest + 0.2 * h))
    
    left_xs = []
    right_xs = []
    valid_ys = []
    
    for y in range(y_start, y_end):
        if y >= h:
            break
        nz = np.flatnonzero(mask[y])
        if nz.size > 0:
            left_xs.append(nz[0])
            right_xs.append(nz[-1])
            valid_ys.append(y)
    
    if not left_xs:
        return (int(0.2*w), h//2), (int(0.8*w), h//2)
    
    left_x = int(np.min(left_xs))
    right_x = int(np.max(right_xs))
    
    return (left_x, y_widest), (right_x, y_widest)

def front_src_landmarks(front_rgba):
    a = (front_rgba[...,3] > 0).astype(np.uint8)
    (xL, yL), (xR, yR) = _alpha_band_edges(a, PIVOT_Y_LOW, PIVOT_Y_HIGH)
    return (xL, yL), (xR, yR)

def front_outermost_edges(front_rgba):
    a = (front_rgba[...,3] > 0).astype(np.uint8)
    (xL, yL), (xR, yR) = _get_outermost_edges(a)
    return (xL, yL), (xR, yR)

def _robust_inner_edge_pivot(img_rgba, side='L', flipped=False):
    h, w = img_rgba.shape[:2]
    a = img_rgba[...,3] > 0
    xL = np.full(h, np.nan); xR = np.full(h, np.nan)
    for y in range(h):
        nz = np.flatnonzero(a[y])
        if nz.size:
            xL[y] = nz[0]; xR[y] = nz[-1]
    inner_is_right = (side == 'L') ^ bool(flipped)
    x_inner = xR if inner_is_right else xL
    y0 = max(0, min(h-1, int(round(PIVOT_Y_LOW*h))))
    y1 = max(0, min(h-1, int(round(PIVOT_Y_HIGH*h))))
    if y1 <= y0: y0, y1 = 0, h-1
    xs = x_inner[y0:y1+1]; m = ~np.isnan(xs)
    if not np.any(m):
        xs = x_inner; m = ~np.isnan(xs)
    if not np.any(m):
        return int(0.5*w), int(0.5*h)
    x = int(np.median(xs[m])); y = int(0.5*(y0+y1))
    return x, y

def pivot_from_alpha(img_rgba, side='L', flipped=False):
    x, y = _robust_inner_edge_pivot(img_rgba, side, flipped)
    w = img_rgba.shape[1]
    eps = max(1, int(round(0.003 * w)))
    half = 0.5 * eps
    p0 = (int(round(x - half)), y)
    p1 = (int(round(x + half)), y)
    return (p0, p1)

def _path_for(fid, part, size_label):
    p1 = os.path.join(FRAME_DIR, f"Frame_{fid}_{part}_{size_label}.png")
    p2 = os.path.join(FRAME_DIR, f"Frame_{fid}_{part}.png")
    return p1 if os.path.exists(p1) else p2

def load_layers(fid, size_label):
    front_p = _path_for(fid, "front", size_label)
    left_p = _path_for(fid, "left_arm", size_label)
    right_p = _path_for(fid, "right_arm", size_label)
    
    print(f"[DEBUG] Loading Frame {fid}, Size {size_label}")
    print(f"  Front: {front_p} - exists: {os.path.exists(front_p)}")
    print(f"  Left:  {left_p} - exists: {os.path.exists(left_p) if left_p else False}")
    print(f"  Right: {right_p} - exists: {os.path.exists(right_p) if right_p else False}")
    
    def load_and_optimize(path, max_size=512):
        img = _read_rgba(path)
        h, w = img.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > max_size:
            scale = max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"    [RESIZE] {w}x{h} -> {new_w}x{new_h} (scale={scale:.3f})")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    
    layers = {"front": load_and_optimize(front_p, max_size=512)}
    
    L = load_and_optimize(left_p, max_size=512) if left_p and os.path.exists(left_p) else None
    R = load_and_optimize(right_p, max_size=512) if right_p and os.path.exists(right_p) else None
    
    if L is not None:
        TEST_FLIP_L = int(os.environ.get("TEST_FLIP_L", "2"))  
        
        if TEST_FLIP_L == 1 or ARM_FLIP_X_L:
            L = cv2.flip(L, 1)  
        elif TEST_FLIP_L == 2:
            L = cv2.flip(L, 0)  
        elif TEST_FLIP_L == 3:
            L = cv2.flip(L, -1)  
        
        SRC_L_PIVOT = pivot_from_alpha(L, side='L', flipped=(TEST_FLIP_L in [1,3] or ARM_FLIP_X_L))
        layers["left"] = L
        layers["_src_L_pivot"] = SRC_L_PIVOT
    else:
        layers["_src_L_pivot"] = None
    
    if R is not None:
        TEST_FLIP_R = int(os.environ.get("TEST_FLIP_R", "2"))  
        
        if TEST_FLIP_R == 1 or ARM_FLIP_X_R:
            R = cv2.flip(R, 1)  
        elif TEST_FLIP_R == 2:
            R = cv2.flip(R, 0)  
        elif TEST_FLIP_R == 3:
            R = cv2.flip(R, -1)  
        
        SRC_R_PIVOT = pivot_from_alpha(R, side='R', flipped=(TEST_FLIP_R in [1,3] or ARM_FLIP_X_R))
        layers["right"] = R
        layers["_src_R_pivot"] = SRC_R_PIVOT
    else:
        layers["_src_R_pivot"] = None
    
    FRONT_SRC_PTS = front_src_landmarks(layers["front"])
    layers["_front_src_pts"] = FRONT_SRC_PTS
    
    FRONT_OUTERMOST_PTS = front_outermost_edges(layers["front"])
    layers["_front_outermost_pts"] = FRONT_OUTERMOST_PTS
    
    # Optimize Calculation of Eyeglass Arm Information Using Caching
    if L is not None and "_src_L_pivot" in layers and layers["_src_L_pivot"] is not None:
        SRC_L_PIVOT = layers["_src_L_pivot"]

        temple_id_L = f"left_{fid}_{size_label}"
        temple_info_L = _temple_cache.get_or_compute_temple_info(temple_id_L, L, SRC_L_PIVOT)

        front_L = temple_info_L['front_point']
        back_L = temple_info_L['back_point']
        layers["_left_endpoints"] = (front_L, back_L)
        # Store complete cache data for rapid rendering
        layers["_left_temple_info"] = temple_info_L
        print(f"  [LEFT] Temple endpoints: front=({front_L[0]:.1f},{front_L[1]:.1f}) back=({back_L[0]:.1f},{back_L[1]:.1f})")
    else:
        layers["_left_endpoints"] = None
        layers["_left_temple_info"] = None
    
    if R is not None and "_src_R_pivot" in layers and layers["_src_R_pivot"] is not None:
        SRC_R_PIVOT = layers["_src_R_pivot"]
       
        temple_id_R = f"right_{fid}_{size_label}"
        temple_info_R = _temple_cache.get_or_compute_temple_info(temple_id_R, R, SRC_R_PIVOT)
        
        front_R = temple_info_R['front_point']
        back_R = temple_info_R['back_point']
        layers["_right_endpoints"] = (front_R, back_R)
       
        layers["_right_temple_info"] = temple_info_R
        print(f"  [RIGHT] Temple endpoints: front=({front_R[0]:.1f},{front_R[1]:.1f}) back=({back_R[0]:.1f},{back_R[1]:.1f})")
    else:
        layers["_right_endpoints"] = None
        layers["_right_temple_info"] = None
    
    return layers

def default_state(fid, size_label):
    return {
        "frameId": fid,
        "size": size_label,
        "fitMode": "manual",
        "fitDone": True,
    }

layers = {}
state_lock = threading.Lock()
STATE = default_state(DEFAULT_FRAME_ID, DEFAULT_SIZE)
smooth = {}
last_jpeg = None
last_ts = 0


ipd_stabilizer = None

def _apply_size_and_reload(sz):
    with state_lock:
        STATE["size"] = sz
        fid = STATE.get("frameId", DEFAULT_FRAME_ID)
    # Clear the arm cache when switching sizes
    _temple_cache.clear_cache()
    global layers
    layers = load_layers(fid, sz)
    print(f"[INFO] Loaded frame {fid} with size {sz}")

def _switch_frame(fid):
    with state_lock:
        STATE["frameId"] = fid
        sz = STATE["size"]
  
    _temple_cache.clear_cache()
    global layers
    layers = load_layers(fid, sz)
    for k in smooth:
        smooth[k] = None
    print(f"[INFO] Switched to frame {fid}")

def warp_rgba(img_rgba, src_pivot, dst_px, u, n, d, out_shape,
              scale=1.0, sx=1.0, sy=1.0, theta_add_deg=0.0, base_dist_ref=100.0):
    
    (p0_src, p1_src) = src_pivot
    p_center_src = np.array(p0_src, np.float32)
    
    # Calculate scaling
    ref = float(base_dist_ref)
    sx_final = (d / ref) * float(scale) * float(sx)
    sy_final = (d / ref) * float(scale) * float(sy)
    
    # Calculate the rotation angle
    theta_base = math.atan2(float(u[1]), float(u[0]))
    theta_total = theta_base + math.radians(theta_add_deg)
    c, s = math.cos(theta_total), math.sin(theta_total)
    
    # Construct the transformation matrix
    M2 = np.array([[c, -s], [s, c]], np.float32) @ np.array([[sx_final, 0], [0, sy_final]], np.float32)
    t = dst_px - (M2 @ p_center_src)
    

    M = np.zeros((2, 3), np.float32)
    M[:, :2] = M2
    M[:, 2] = t
    
    H_out, W_out = out_shape[:2]
    warped = cv2.warpAffine(img_rgba, M, (W_out, H_out),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0,0,0,0))
    
    bgr = warped[..., :3].astype(np.float32)
    alpha = warped[..., 3:4].astype(np.float32) / 255.0
    return bgr, alpha

def generate_stream():
    global last_jpeg, last_ts, smooth, ipd_stabilizer
    # cap = cv2.VideoCapture(CAM_INDEX)
    cap = AsyncVideoCapture(src=CAM_INDEX, width=W_CAP, height=H_CAP)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W_CAP)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_CAP)
    
    # Initialize SmartFrameFaceMatcher
    verify_database_paths()  
    smart_matcher = SmartFrameFaceMatcher(
        face_db_path=FACE_DB_PATH,
        eye_db_path=EYE_DB_PATH
    )
    
    # Global ipd_stabilizer (for reset)
    if ipd_stabilizer is None:
        ipd_stabilizer = IPDStabilizer(
            window_size=IPD_SMOOTH_WINDOW,
            yaw_threshold=IPD_YAW_THRESHOLD,
            yaw_max=IPD_YAW_MAX,
            min_ratio=IPD_MIN_RATIO
        )
    
    size_label = DEFAULT_SIZE
    global layers
    if not layers:
        with state_lock:
            fid = STATE.get("frameId", DEFAULT_FRAME_ID)
        layers = load_layers(fid, size_label)
    
    fps_ema = 0.0
    t_prev = time.perf_counter()
    jpg_quality = 80
    
    FRONT_BASE_DIST = 100.0
    
    # Performance Analysis
    perf_counter = 0
    perf_times = {'cap': [], 'mediapipe': [], 'render': [], 'jpeg': [], 'total': []}
    
    while True:
        # frame start
        t_frame_start = time.perf_counter()
        
        # Measuring Camera Read Performance
        t1 = time.perf_counter()
        ok_cap, frame = cap.read()
        t2 = time.perf_counter()
        t_cap = (t2 - t1) * 1000
        if not ok_cap:
            time.sleep(0.01)
            continue
        
        if H_MIRROR:
            frame = cv2.flip(frame, 1)
        
        H, W = frame.shape[:2]
        
        # Measuring MediaPipe Performance
        t3 = time.perf_counter()
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t4 = time.perf_counter()
        t_mp = (t4 - t3) * 1000
        
        # Rendering Markup
        has_face = False
        t_render_start = time.perf_counter()
        
        if res.multi_face_landmarks:
            has_face = True
            lm = res.multi_face_landmarks[0].landmark
            yaw_rad, pitch_rad, roll_rad = estimate_head_pose(lm, W, H)
            
            yaw_deg = math.degrees(yaw_rad)
            yaw_abs = abs(yaw_deg)
            
            if LANDMARK_MODE == "stable":
                pL_img, pR_img, indicesL, indicesR, ipd = get_glasses_anchor_points_stable(
                    lm, W, H, yaw_rad, ipd_stabilizer
                )
            else:
                L_outer = lm[33]; R_outer = lm[263]
                pL_img = to_img_px(L_outer, W, H)
                pR_img = to_img_px(R_outer, W, H)
                indicesL = [33]; indicesR = [263]
                ipd = float(np.linalg.norm(pR_img - pL_img))
            
            # Use SmartFrameFaceMatcher to obtain matching parameters
            smart_params = smart_matcher.analyze_and_fit(lm, W, H, ipd)
            
            # Calculate Z-depth
            zL = get_z_for_point(lm, indicesL)
            zR = get_z_for_point(lm, indicesR)
            
            smooth["pL_img"] = pL_img if smooth.get("pL_img") is None else ema(smooth["pL_img"], pL_img, SMOOTH_A)
            smooth["pR_img"] = pR_img if smooth.get("pR_img") is None else ema(smooth["pR_img"], pR_img, SMOOTH_A)
            smooth["yaw"] = yaw_rad if smooth.get("yaw") is None else smooth_angle(smooth["yaw"], yaw_rad, SMOOTH_A)
            smooth["pitch"] = pitch_rad if smooth.get("pitch") is None else smooth_angle(smooth["pitch"], pitch_rad, SMOOTH_A)
            smooth["roll"] = roll_rad if smooth.get("roll") is None else smooth_angle(smooth["roll"], roll_rad, SMOOTH_A)
            smooth["ipd"] = ipd if smooth.get("ipd") is None else ema(smooth["ipd"], ipd, SMOOTH_A)
            smooth["zL"] = zL if smooth.get("zL") is None else ema(smooth["zL"], zL, SMOOTH_A)
            smooth["zR"] = zR if smooth.get("zR") is None else ema(smooth["zR"], zR, SMOOTH_A)
            
            pL_s = smooth["pL_img"]
            pR_s = smooth["pR_img"]
            yaw_s = smooth["yaw"]
            pitch_s = smooth["pitch"]
            roll_s = smooth["roll"]
            
            d_img_raw = float(np.linalg.norm(pR_s - pL_s))
            smooth["d_img"] = d_img_raw if smooth.get("d_img") is None else ema(smooth["d_img"], d_img_raw, SMOOTH_A)
            d_img = smooth["d_img"]
            
            
            dz = (smooth.get("zR", 0.0) or 0.0) - (smooth.get("zL", 0.0) or 0.0)
            
            center_px = 0.5 * (pL_s + pR_s)
            
            with state_lock:
                fitMode = STATE.get("fitMode", "manual")
                fitDone = STATE.get("fitDone", True)
                size_label = STATE.get("size", size_label)
                fid = STATE.get("frameId", DEFAULT_FRAME_ID)  
            
            if fitMode == "once" and not fitDone:
                ratio = d_img / W
                if ratio < T_SM:
                    size_label = "S"
                elif ratio < T_ML:
                    size_label = "M"
                else:
                    size_label = "L"
                with state_lock:
                    STATE["size"] = size_label
                    STATE["fitDone"] = True
                _apply_size_and_reload(size_label)
            
            kScale = SIZE_SCALE.get(size_label, 1.0) * MANUAL_SCALE_ADJUST
            
            # Apply Smart Scaling Coefficient
            kScale *= smart_params.get('scale_multiplier', 1.0)
            
            # Apply per-frame scaling adjustments 
            frame_scale_adjust = FRAME_SCALE_ADJUSTMENTS.get(fid, 1.0)
            kScale *= frame_scale_adjust
            
            kSX = smart_params.get('width_factor', 1.0)
            kSY = smart_params.get('height_factor', 1.0)
            
            u_img = pR_s - pL_s
            theta_img = math.atan2(u_img[1], u_img[0])
            
            n_img_perp = np.array([-u_img[1], u_img[0]], np.float32)
            n_img_perp_norm = n_img_perp / (np.linalg.norm(n_img_perp) + 1e-9)
            
            u_img_norm = u_img / (np.linalg.norm(u_img) + 1e-9)
            
            yaw_shear = 0.0
            if abs(yaw_s) > 0.01:
                # Limit the range of tan(yaw) to prevent extreme shear at large angles.
                tan_yaw = math.tan(yaw_s)
                tan_yaw_clamped = max(-0.7, min(0.7, tan_yaw))
                yaw_shear = -SHEAR_K * tan_yaw_clamped
            
            z_shear_comp = np.array([0.0, 0.0], np.float32)
            if USE_Z_SHEAR:
                zL = get_z_for_point(lm, indicesL)
                zR = get_z_for_point(lm, indicesR)
                dz = zR - zL
                if abs(dz) > 0.001:
                    z_shear = Z_SHEAR_K * dz
                    z_shear_comp = n_img_perp_norm * z_shear * (d_img / 2.0)
            
            
            center_base = center_px.copy()
            
            # Applying bias in the u-n coordinate system
            center_base += CENTER_T_BIAS * u_img_norm
            center_base += CENTER_N_BIAS * n_img_perp_norm
            
            shear_vec = yaw_shear * d_img * n_img_perp_norm * 0.3  
            center_final = center_base + shear_vec + z_shear_comp
            
            u_img_s = u_img * kSX
            n_img_s = n_img_perp * kSY
            
            dist_to_center = np.array([0.5 * d_img, MANUAL_OFFSET_Y], np.float32)
            dist_to_center[0] += MANUAL_OFFSET_X
            
            # Vertical Offset with Intelligent Adaptation
            smart_vertical_offset = smart_params.get('vertical_offset', 0.0)
            dist_to_center[1] += smart_vertical_offset
            
            # Per-frame vertical offset adjustment
            frame_vertical_adjust = FRAME_VERTICAL_ADJUSTMENTS.get(fid, 0.0)
            dist_to_center[1] += frame_vertical_adjust
            
    
            vertical_offset_corrected = -dist_to_center[1]
            pL_anchor = center_final - dist_to_center[0] * u_img_norm + vertical_offset_corrected * n_img_perp_norm
            pR_anchor = center_final + dist_to_center[0] * u_img_norm + vertical_offset_corrected * n_img_perp_norm
            
            SRC_L, SRC_R = layers.get("_front_src_pts", ((0, 0), (1, 1)))
            p_src_L = np.array(SRC_L, np.float32)
            p_src_R = np.array(SRC_R, np.float32)
            
            p_dst_L = pL_anchor
            p_dst_R = pR_anchor
            
            frame_tilt_deg = smart_params.get('frame_tilt', 0.0)  
            
            u_dst = p_dst_R - p_dst_L
            theta_dst = math.atan2(u_dst[1], u_dst[0]) + math.radians(frame_tilt_deg)
            
        
            d_dst = d_img  
            d_src_front = float(np.linalg.norm(p_src_R - p_src_L))
            
            front_scale_val = (d_dst / (d_src_front + 1e-9)) * kScale
            sx_scale = kSX
            
            cos_dst = math.cos(theta_dst)
            sin_dst = math.sin(theta_dst)
            
            s_x = front_scale_val * sx_scale
            s_y = front_scale_val * FRAME_HEIGHT_SCALE * kSY
            
            R_front = np.array([
                [s_x * cos_dst, -s_y * sin_dst],
                [s_x * sin_dst,  s_y * cos_dst]
            ], np.float32)
            
            c_dst = 0.5 * (p_dst_L + p_dst_R)
            c_src_front = 0.5 * (p_src_L + p_src_R)
            
            c_dst_col = c_dst.reshape(2, 1)
            c_src_col = c_src_front.reshape(2, 1)
            t_front = c_dst_col - R_front @ c_src_col
            
            M_front = np.hstack([R_front, t_front])
            
            fg_front = cv2.warpAffine(
                layers["front"][..., :3],
                M_front, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,0,0)
            )
            a_front_mask = cv2.warpAffine(
                layers["front"][..., 3],
                M_front, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            a_front = (a_front_mask.astype(np.float32) / 255.0)[..., None]
            
           
            SRC_L_PIVOT = layers.get("_src_L_pivot")
            SRC_R_PIVOT = layers.get("_src_R_pivot")
            
            (HINGE_L_SRC, HINGE_R_SRC) = layers.get("_front_outermost_pts", ((0, 0), (1, 1)))
            p_hinge_L_src = np.array(HINGE_L_SRC, np.float32).reshape(2,1)
            p_hinge_R_src = np.array(HINGE_R_SRC, np.float32).reshape(2,1)
            p_hinge_L_dst = (M_front[:, :2] @ p_hinge_L_src + M_front[:, 2:3]).flatten()
            p_hinge_R_dst = (M_front[:, :2] @ p_hinge_R_src + M_front[:, 2:3]).flatten()
            
            with state_lock:
                fid = STATE.get("frameId", DEFAULT_FRAME_ID)
            
            if IPD_STABILIZATION_ENABLED and ipd_stabilizer.baseline_d is not None:
                d_reference = ipd_stabilizer.baseline_d  
            else:
                d_reference = d_img  
            
            BASE_OFFSET_RATIO = 0.50  
            MAX_OFFSET_INCREASE = 0.25  
            ANGLE_THRESHOLD = 30.0  
            ANGLE_MAX = 60.0  
            
            if yaw_abs > ANGLE_THRESHOLD:
                angle_factor = min((yaw_abs - ANGLE_THRESHOLD) / (ANGLE_MAX - ANGLE_THRESHOLD), 1.0)
                offset_ratio = BASE_OFFSET_RATIO + angle_factor * MAX_OFFSET_INCREASE
            else:
                offset_ratio = BASE_OFFSET_RATIO
            
            horizontal_offset = d_reference * offset_ratio  
            vertical_offset = d_reference * 0.12    
            
          
            offset_frame_local_L = np.array([-horizontal_offset, vertical_offset], np.float32)
            offset_frame_local_R = np.array([+horizontal_offset, vertical_offset], np.float32)
            
        
            R_rotation_only = np.array([
                [cos_dst, -sin_dst],
                [sin_dst,  cos_dst]
            ], np.float32)
            
            offset_img_L = R_rotation_only @ offset_frame_local_L.reshape(2, 1)
            offset_img_R = R_rotation_only @ offset_frame_local_R.reshape(2, 1)
            
          
            pL_anchor = p_hinge_L_dst + offset_img_L.flatten()
            pR_anchor = p_hinge_R_dst + offset_img_R.flatten()
            
           
            templeRotDeg = yaw_deg * 0.1
            temple_rotation = smart_params.get('temple_rotation', 0.0)
            templeRotDeg += temple_rotation
            
           
            temple_len = horizontal_offset
            
            
            sx_L = 1.0
            sx_R = 1.0
            
            occlusion_L = 1.0
            occlusion_R = 1.0
            
            if OCCLUSION_ENABLED:
                if yaw_abs < FRONT_HIDE_THRESHOLD:
                    occlusion_L = 1.0
                    occlusion_R = 1.0
                else:
                    if yaw_abs < FRONT_SHOW_THRESHOLD:
                        fade_ratio = (yaw_abs - FRONT_HIDE_THRESHOLD) / (FRONT_SHOW_THRESHOLD - FRONT_HIDE_THRESHOLD)
                        occlusion_L = 1.0
                        occlusion_R = 1.0
                    else:
                        if yaw_abs < SIDE_HIDE_START:
                            occlusion_L = 1.0
                            occlusion_R = 1.0
                        elif yaw_abs < SIDE_HIDE_COMPLETE:
                            hide_ratio = (yaw_abs - SIDE_HIDE_START) / (SIDE_HIDE_COMPLETE - SIDE_HIDE_START)
                            
                            if yaw_deg > 0:
                                occlusion_L = 1.0
                                occlusion_R = max(0.0, 1.0 - hide_ratio * 1.5)
                            else:
                                occlusion_L = max(0.0, 1.0 - hide_ratio * 1.5)
                                occlusion_R = 1.0
                                
                        else:
                            if yaw_deg > 0:
                                occlusion_L = 1.0
                                occlusion_R = 0.0
                            else:
                                occlusion_L = 0.0
                                occlusion_R = 1.0
            else:
                occlusion_L = 1.0
                occlusion_R = 1.0
            
            temple_stretch = 1.0
            if yaw_abs > 30:
                stretch_factor = min((yaw_abs - 30) / 30, 1.0)
                temple_stretch = 1.0 + stretch_factor * 0.8
                
                sx_L *= temple_stretch
                sx_R *= temple_stretch
            
            smooth["temple_stretch"] = temple_stretch
            
            d_for_temple = d_img  
            
            arm_scale = kScale * ARM_SCALE_MULTIPLIER
            
            # Symmetrical left and right temple arms
            temple_angle_L = templeRotDeg + PNG_TILT_COMPENSATION + TEMPLE_ANGLE_OFFSET_L  
            temple_angle_R = templeRotDeg + PNG_TILT_COMPENSATION + TEMPLE_ANGLE_OFFSET_R  
            
            if OCCLUSION_ENABLED and yaw_deg > FRONT_SHOW_THRESHOLD:
                render_order = ["right", "left"]
            elif OCCLUSION_ENABLED and yaw_deg < -FRONT_SHOW_THRESHOLD:
                render_order = ["left", "right"]
            else:
                render_order = ["left", "right"]
            

            # HUD display 
            # if SHOW_HUD:
            #     H, W = frame.shape[:2]
            #     cv2.circle(frame, tuple(np.int32(p_hinge_L_dst)), 4, (255, 0, 255), -1) 
            #     cv2.circle(frame, tuple(np.int32(p_hinge_R_dst)), 4, (255, 0, 255), -1)
            #     
            #     
            #     if "_left_endpoints" in layers and "_right_endpoints" in layers:
            #         cv2.circle(frame, tuple(np.int32(p_hinge_L_dst)), 3, (255, 128, 0), -1)  
            #         cv2.circle(frame, tuple(np.int32(p_hinge_R_dst)), 3, (255, 128, 0), -1)  
            #     # anchor
            #     cv2.circle(frame, tuple(np.int32(pL_anchor)), 6, (0, 255, 255), 2)
            #     cv2.circle(frame, tuple(np.int32(pR_anchor)), 6, (0, 255, 255), 2)
            #     # Line
            #     cv2.line(frame, tuple(np.int32(p_hinge_L_dst)), tuple(np.int32(pL_anchor)), (255, 255, 0), 1)
            #     cv2.line(frame, tuple(np.int32(p_hinge_R_dst)), tuple(np.int32(pR_anchor)), (255, 255, 0), 1)
            #     # Info Display
            #     info_y = H - 60
            #     cv2.putText(frame, f"Temple L:({int(pL_anchor[0])},{int(pL_anchor[1])})", 
            #                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
            #     cv2.putText(frame, f"Temple R:({int(pR_anchor[0])},{int(pR_anchor[1])})",
            #                (10, info_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

            #     cv2.putText(frame, f"offset={horizontal_offset:.1f}px ratio={offset_ratio:.3f}",
            #                (10, info_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,128), 1)

            #     d_ref_str = f"d_ref={d_reference:.1f}px"
            #     if IPD_STABILIZATION_ENABLED and ipd_stabilizer.baseline_d is not None:
            #         d_ref_str += f" (baseline)"
            #     cv2.putText(frame, d_ref_str,
            #                (10, info_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,128), 1)
            #     cv2.putText(frame, f"v40B: Frame{fid} scale={frame_scale_adjust:.2f} vOffset={frame_vertical_adjust:.1f}px",
            #                (10, info_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,255,0), 1)

            
            # Front-view arm cover (improved, see lines 944-951 of the code)
            yaw_abs = abs(yaw_deg)
            if FRONT_TEMPLE_FADEOUT_ENABLED:
                if yaw_abs < FRONT_FADEOUT_START: 
                    temple_alpha = 0.0  
                elif yaw_abs < FRONT_FADEOUT_END:  
                    temple_alpha = (yaw_abs - FRONT_FADEOUT_START) / (FRONT_FADEOUT_END - FRONT_FADEOUT_START)  
                else:  
                    temple_alpha = 1.0 
            else:
                temple_alpha = 1.0
            
            
            # Use the hinge point directly as the binding point
            dst_front_L_adjusted = p_hinge_L_dst  
            dst_front_R_adjusted = p_hinge_R_dst  
            
            # Adaptive Fade Parameter Calculation
            if ADAPTIVE_FADEOUT_ENABLED and TEMPLE_FADEOUT_ENABLED:
                if yaw_abs < ADAPTIVE_YAW_START:
                    adaptive_fadeout_start = FRONT_FADEOUT_START_RATIO
                    adaptive_fadeout_end = FRONT_FADEOUT_END_RATIO
                elif yaw_abs < ADAPTIVE_YAW_END:
                    transition_ratio = (yaw_abs - ADAPTIVE_YAW_START) / (ADAPTIVE_YAW_END - ADAPTIVE_YAW_START)
                    adaptive_fadeout_start = FRONT_FADEOUT_START_RATIO + \
                        (SIDE_FADEOUT_START_RATIO - FRONT_FADEOUT_START_RATIO) * transition_ratio
                    adaptive_fadeout_end = FRONT_FADEOUT_END_RATIO + \
                        (SIDE_FADEOUT_END_RATIO - FRONT_FADEOUT_END_RATIO) * transition_ratio
                else:
                    adaptive_fadeout_start = SIDE_FADEOUT_START_RATIO
                    adaptive_fadeout_end = SIDE_FADEOUT_END_RATIO
            else:
                adaptive_fadeout_start = TEMPLE_FADEOUT_START
                adaptive_fadeout_end = TEMPLE_FADEOUT_END
            
            # Generate facial contour mask
            face_mask = None
            if FACE_MASK_ENABLED:
                face_mask = create_face_mask(lm, frame.shape, 
                                            expand=FACE_MASK_EXPAND, 
                                            feather=FACE_MASK_FEATHER)
            
            for side in render_order:
                if side == "left" and "left" in layers:
                    if "_left_temple_info" in layers and layers["_left_temple_info"] is not None:
                        temple_info_L = layers["_left_temple_info"]
                        fg_L, a_L = warp_temple_with_skinning_fast(
                            layers["left"],
                            temple_info=temple_info_L,          
                            dst_front=pL_anchor,                 
                            dst_back=dst_front_L_adjusted,       
                            out_shape=frame.shape,
                            width_scale=TEMPLE_WIDTH_SCALE_L    
                        )
                        if TEMPLE_FADEOUT_ENABLED:
                            a_L = apply_temple_fadeout_fast(
                                a_L,
                                hinge_point=dst_front_L_adjusted,
                                anchor_point=pL_anchor,
                                fade_start_ratio=adaptive_fadeout_start,  
                                fade_end_ratio=adaptive_fadeout_end        
                            )
                    elif "_left_endpoints" in layers:
                        src_front_L, src_back_L = layers["_left_endpoints"]
                        fg_L, a_L = warp_temple_with_skinning(
                            layers["left"],
                            src_front=src_front_L,
                            src_back=src_back_L,
                            dst_front=pL_anchor,
                            dst_back=dst_front_L_adjusted,
                            out_shape=frame.shape,
                            width_scale=TEMPLE_WIDTH_SCALE_L
                        )
                        if TEMPLE_FADEOUT_ENABLED:
                            a_L = apply_temple_fadeout(
                                a_L,
                                hinge_point=dst_front_L_adjusted,
                                anchor_point=pL_anchor,
                                fade_start_ratio=adaptive_fadeout_start,  
                                fade_end_ratio=adaptive_fadeout_end,      
                                frame_shape=frame.shape
                            )
                    else:
                        fg_L, a_L = warp_rgba(
                            layers["left"], SRC_L_PIVOT,
                            pL_anchor, u_img_s, n_img_s, d_for_temple, frame.shape,
                            scale=arm_scale, sx=kSX * sx_L, sy=kSY,
                            theta_add_deg=temple_angle_L,
                            base_dist_ref=FRONT_BASE_DIST
                        )
                    a_L_final = (a_L * occlusion_L if OCCLUSION_ENABLED else a_L) * temple_alpha  
                    
                    if face_mask is not None:
                        if a_L_final.ndim == 3:
                            face_mask_3d = face_mask[:, :, np.newaxis]
                            a_L_final = a_L_final * face_mask_3d
                        else:
                            a_L_final = a_L_final * face_mask
                    
                    frame = (frame * (1 - a_L_final) + fg_L * a_L_final).astype(np.uint8)
                
                elif side == "right" and "right" in layers:
                    if "_right_temple_info" in layers and layers["_right_temple_info"] is not None:
                        temple_info_R = layers["_right_temple_info"]
                        fg_R, a_R = warp_temple_with_skinning_fast(
                            layers["right"],
                            temple_info=temple_info_R,           
                            dst_front=pR_anchor,                 
                            dst_back=dst_front_R_adjusted,       
                            out_shape=frame.shape,
                            width_scale=TEMPLE_WIDTH_SCALE_R     
                        )
                        if TEMPLE_FADEOUT_ENABLED:
                            a_R = apply_temple_fadeout_fast(
                                a_R,
                                hinge_point=dst_front_R_adjusted,
                                anchor_point=pR_anchor,
                                fade_start_ratio=adaptive_fadeout_start,  
                                fade_end_ratio=adaptive_fadeout_end        
                            )
                    elif "_right_endpoints" in layers:
                        src_front_R, src_back_R = layers["_right_endpoints"]
                        fg_R, a_R = warp_temple_with_skinning(
                            layers["right"],
                            src_front=src_front_R,
                            src_back=src_back_R,
                            dst_front=pR_anchor,
                            dst_back=dst_front_R_adjusted,
                            out_shape=frame.shape,
                            width_scale=TEMPLE_WIDTH_SCALE_R
                        )
                        if TEMPLE_FADEOUT_ENABLED:
                            a_R = apply_temple_fadeout(
                                a_R,
                                hinge_point=dst_front_R_adjusted,
                                anchor_point=pR_anchor,
                                fade_start_ratio=adaptive_fadeout_start,  
                                fade_end_ratio=adaptive_fadeout_end,      
                                frame_shape=frame.shape
                            )
                    else:
                        fg_R, a_R = warp_rgba(
                            layers["right"], SRC_R_PIVOT,
                            pR_anchor, u_img_s, n_img_s, d_for_temple, frame.shape,
                            scale=arm_scale, sx=kSX * sx_R, sy=kSY,
                            theta_add_deg=temple_angle_R,
                            base_dist_ref=FRONT_BASE_DIST
                        )
                    a_R_final = (a_R * occlusion_R if OCCLUSION_ENABLED else a_R) * temple_alpha  
                    
                    
                    if face_mask is not None:
                        if a_R_final.ndim == 3:
                            face_mask_3d = face_mask[:, :, np.newaxis]
                            a_R_final = a_R_final * face_mask_3d
                        else:
                            a_R_final = a_R_final * face_mask
                    
                    frame = (frame * (1 - a_R_final) + fg_R * a_R_final).astype(np.uint8)
            
            frame = (frame * (1 - a_front) + fg_front * a_front).astype(np.uint8)

            # HUD display
            # if SHOW_HUD:
            #     # Ensure that the smooth value is not None
            #     if smooth.get("pL_img") is not None and smooth.get("pR_img") is not None:
            #         cv2.circle(frame, tuple(np.int32(smooth["pL_img"])), 4, (0,255,0), -1)
            #         cv2.circle(frame, tuple(np.int32(smooth["pR_img"])), 4, (0,0,255), -1)
            #         cv2.circle(frame, tuple(np.int32(pL_anchor)), 3, (255,255,0), -1)
            #         cv2.circle(frame, tuple(np.int32(pR_anchor)), 3, (0,255,255), -1)
            #     
            #     has_left = "left" in layers
            #     has_right = "right" in layers
            #     arms_status = f"L:{has_left} R:{has_right}"
            #     
            #     # Info display
            #     debug_info = smart_params.get('_debug', {})
            #     face_shape_d = debug_info.get('face_shape', 'N/A')
            #     eye_shape_d = debug_info.get('eye_shape', 'N/A')
            #     locked = debug_info.get('locked', False)
            #     smart_text = f"Face:{face_shape_d} Eye:{eye_shape_d} {'LOCKED' if locked else ''}"
            #     cv2.putText(frame, smart_text, (14, 156),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,255,128), 2, cv2.LINE_AA)
            #     
            #     if IPD_STABILIZATION_ENABLED and ipd_stabilizer.baseline_ipd is not None:
            #         baseline_text = f"IPD_base={ipd_stabilizer.baseline_ipd:.1f}px"
            #         if ipd_stabilizer.baseline_d is not None:
            #             baseline_text += f" d_base={ipd_stabilizer.baseline_d:.1f}px"
            #         temple_stretch_val = smooth.get("temple_stretch", 1.0) or 1.0
            #         if temple_stretch_val > 1.01:
            #             baseline_text += f" stretch={temple_stretch_val:.2f}x"
            #         cv2.putText(frame, baseline_text, (14, 126),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,128,255), 2, cv2.LINE_AA)
            #     
            #     cv2.putText(frame, f"Size:{size_label} {arms_status}", (14, 36),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
            #     d_img_val = smooth.get('d_img', 0) or 0
            #     ipd_val = smooth.get('ipd', 0) or 0
            #     cv2.putText(frame, f"d={d_img_val:.1f}px sx={sx_scale:.3f}", (14, 66),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2, cv2.LINE_AA)
            #     cv2.putText(frame, f"ipd={ipd_val:.1f}px yaw={yaw_deg:+.1f} dz={dz:.3f}", (14, 96),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,200,255), 2, cv2.LINE_AA)
        
        # Measure rendering time
        if has_face:
            t_render_end = time.perf_counter()
            t_render = (t_render_end - t_render_start) * 1000
        else:
            t_render = 0

        t = time.perf_counter()
        dt = t - t_prev
        t_prev = t
        if dt > 0:
            fps = 1.0 / dt
            fps_ema = fps if fps_ema == 0 else 0.9*fps_ema + 0.1*fps
            cv2.putText(frame, f"{fps_ema:4.1f} FPS", (W-200, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        
        # Measuring JPEG Encoding Performance
        t5 = time.perf_counter()
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        t6 = time.perf_counter()
        t_jpeg = (t6 - t5) * 1000
        
        # Total Frame Time
        t_frame_end = time.perf_counter()
        t_total = (t_frame_end - t_frame_start) * 1000
        
        # Collect data
        perf_times['cap'].append(t_cap)
        perf_times['mediapipe'].append(t_mp)
        perf_times['render'].append(t_render)
        perf_times['jpeg'].append(t_jpeg)
        perf_times['total'].append(t_total)
        
        # Print the average data every 60 frames
        perf_counter += 1
        if perf_counter % 60 == 0:
            import statistics
            print(f"\n{'='*60}")
            print(f"Performance Analysis (Average of the past 60 frames, Frame #{perf_counter})")
            print(f"{'='*60}")
            print(f"{'Camera readout':<20s}: {statistics.mean(perf_times['cap']):6.2f}ms")
            print(f"{'MediaPipe Detection':<20s}: {statistics.mean(perf_times['mediapipe']):6.2f}ms")
            print(f"{'Rendering Pipeline':<20s}: {statistics.mean(perf_times['render']):6.2f}ms")
            print(f"{'JPEG encoding':<20s}: {statistics.mean(perf_times['jpeg']):6.2f}ms")
            print(f"{'-'*60}")
            avg_total = statistics.mean(perf_times['total'])
            print(f"{'Average Frame Time':<20s}: {avg_total:6.2f}ms")
            print(f"{'Theoretical FPS':<20s}: {1000/avg_total:6.2f}")
            print(f"{'Actual displayed FPS':<20s}: {fps_ema:6.2f}  ← Actual HUD display")
            print(f"{'='*60}\n")
            # Clear list
            for key in perf_times:
                perf_times[key] = []

        if not ok: continue
        last_jpeg, last_ts = buf.tobytes(), time.time()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_jpeg + b'\r\n')

app = Flask(__name__, static_folder=".", static_url_path="")

@app.route("/")
def root():
    return send_from_directory(".", "glasses_virtual_try_on.html")

@app.route("/stream.mjpg")
def stream_jpg():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshot")
def snapshot():
    if last_jpeg is None: return "no frame yet", 503
    return Response(last_jpeg, headers={
        "Content-Type": "image/jpeg",
        "Content-Disposition": f'attachment; filename=\"snapshot_{int(last_ts)}.jpg\"'
    })

@app.route("/api/size", methods=["POST"])
def api_size():
    data = request.get_json(force=True) or {}
    sz = (data.get("size") or "").upper()
    if sz not in SIZES:
        return jsonify(ok=False, err="size must be S/M/L"), 400
    with state_lock:
        STATE["fitMode"] = "manual"
        STATE["fitDone"] = True
    _apply_size_and_reload(sz)
    with state_lock:
        cur = dict(STATE)
    return jsonify(ok=True, state=cur)

@app.route("/api/switch_frame", methods=["POST"])
def api_switch_frame():
    data = request.get_json(force=True) or {}
    fid = (data.get("frameId") or data.get("id") or data.get("frame") or "").upper()
    if fid not in FRAME_IDS:
        return jsonify(ok=False, err="unknown frame id"), 400
    _switch_frame(fid)
    return jsonify(ok=True, frameId=fid)

@app.route("/api/fit_once", methods=["POST"])
def api_fit_once():
    with state_lock:
        STATE["fitMode"] = "once"
        STATE["fitDone"] = False
    return jsonify(ok=True, state=STATE)

@app.route("/api/reset", methods=["POST"])
def api_reset():
    global ipd_stabilizer
    with state_lock:
        fid = STATE.get("frameId", DEFAULT_FRAME_ID)
        STATE.clear()
        STATE.update(default_state(fid, DEFAULT_SIZE))
    _apply_size_and_reload(DEFAULT_SIZE)
    for k in smooth:
        smooth[k] = None
    if ipd_stabilizer is not None:
        ipd_stabilizer.reset()
    return jsonify(ok=True, state=STATE)

@app.route("/api/debug_layers")
def api_debug_layers():
    info = {}
    for k, img in layers.items():
        if k.startswith("_"): 
            info[k] = img
            continue
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1
        info[k] = {
            "shape": [h, w, c],
            "loaded": True
        }
    return jsonify(info)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AR Virtual Try-On System - Integrated Version")
    print("="*70)
    print("\n[INFO] System features:")
    print("  ✅ IPD Stabilization (prevents size shrinking during head rotation)")
    print("  ✅ Smart Frame-Face Matching (auto-adjusts based on face/eye shape)")
    print("  ✅ Temple arm stretching (realistic side view)")
    print("  ✅ Occlusion handling (natural appearance)")
    print("\n[INFO] Configuration:")
    print(f"  - Face DB: {FACE_DB_PATH if FACE_DB_PATH else 'Not configured (using defaults)'}")
    print(f"  - Eye DB: {EYE_DB_PATH if EYE_DB_PATH else 'Not configured (using defaults)'}")
    print(f"  - Camera: {CAM_INDEX}, Resolution: {W_CAP}x{H_CAP}")
    print(f"  - IPD Stabilization: {'Enabled' if IPD_STABILIZATION_ENABLED else 'Disabled'}")
    print("="*70 + "\n")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False, threaded=True)