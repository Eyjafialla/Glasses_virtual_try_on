# Glasses Virtual Try-On System

## 1.Overview
This project establishes separate databases for various facial shapes and eye shapes, classifying them to provide the basis for real-time camera detection of facial contours. It then uses these prediction results to match different frame sizes, enabling intelligent virtual glasses try-on.

---

## 2.Features
- Face shape prediction
- Eye shape prediction
- Auto fit with the frame size
- 2D Frame rendering
- Real-time glasses virtual try on
- Web AR demo
---

## 3.Project Structure
```text
Glasses_virtual_try_on/
│
├── database/                   # Shape classification data
│   ├── eye_db/
│   │   └── eyes_by_shape.csv
│   └── face_db/
│       └── face_by_shape.csv
│
├── Demo/                       # Demo video output
│   └── Web AR Try-On_demo.mp4
│
├── frames/                     # Glasses frame components (A/B/C/D series)
│   ├── Frame_A_front_L.png
│   ├── Frame_A_left_arm_L.png
│   ├── Frame_A_right_arm_L.png
│   └── ... (More components)
│
├──Frame_A.png                  #Thumbnail images
├──Frame_B.png
├──Frame_C.png
├──Frame_D.png
├── camera_async.py             # Realtime camera try-on script
├── glasses_virtual_try_on.py   # Main processing pipeline
├── glasses_virtual_try_on.html # Web AR interface
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


---

## 4.Installation

### 4.1 Clone the Repository
```bash
git clone https://github.com/Eyjafialla/Glasses_virtual_try_on.git
cd Glasses_virtual_try_on
```

### 4.2 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5.How to run

### 5.1 Run the main program
```bash
python glasses_virtual_try_on.py 
```

### 5.2 Open the web link (chrome recommended)
1.Double click glasses_virtual_try_on.html

2.Open with http://127.0.0.1:5000 or http://192.168.88.4:5000 (ctrl+click)

---

## 6.Frame Rendering 
Rendering Architecture Overview
This system employs a layered rendering architecture, dividing the glasses into three independent layers:

Front Frame Layer
Left Temple Layer
Right Temple Layer

Each layer uses RGBA PNG images with alpha channels, achieving realistic 3D wearing effects through 2D affine transformations and advanced alpha blending techniques.

Front Frame Layer:
1.Anchor point positioning
2.Transformation matrix
3.Perspective distortion

Temple Layer:
1.Skinning Deformation Technique
2.Hinge and anchor point positioning
3.Temple Stretching

Advanced Rendering Features:
1.Adaptive Fadeout
2.Facial Occlusion Mask

Performance optimization:
1.Temple Information Cache
2.Image preprocessing
3.Asynchronous camera readout

---

## 7.Face and Eye Shape Prediction
Using a multi-classifier, the original dataset is categorized into different face shapes/eye shapes.

Face shapes include: round/oval/oblong/square/heart
Eye shapes include: almond_downturned_open/almond_neutral_open/almond_upturned_open/round_upturned_open

Detected faces undergo multi-level classification using a decision tree-like architecture to generate a more accurate prediction.

---

## 8.Configuration

### 8.1 FaceShapeClassifier
```python
self.thresholds = {
    'round_max': 0.975,      # Upper limit for round faces: L/W < 0.975
    'oval_min': 0.88,        # Lower limit for oval faces: L/W >= 0.88
    'oval_max': 1.15,        # Upper limit for oval faces: L/W <= 1.15
    'oblong_min': 1.20,      # Lower limit for long faces: L/W > 1.20
    
    # Secondary Classifiers
    'heart_square_boundary': 1.05,  # Heart-Shaped/Square Divider
    
    # Jaw-Cheek Ratio
    'square_jaw_min': 0.95,   # Square Face: Jaw width ≥ 95% of cheek width
    'heart_jaw_max': 0.85,    # Heart-shaped face: Jaw width ≤ 85% of cheek width
    
    # Jaw-Forehead Ratio
    'heart_jaw_forehead_max': 0.90,  # Heart-shaped face: The jawline is narrower than the forehead.
    'square_jaw_forehead_min': 0.85, # Square face: The jawline is not particularly narrow.
}   
```

### 8.2 EyeShapeClassifier
```python
self.almond_aspect_ratio = (4.5, 7.0)  
self.round_aspect_ratio = (3.0, 5.0)
almond_min = np.percentile(almond_arr, 5)   
almond_max = np.percentile(almond_arr, 95)  

avg_tilt > 2.0°   
avg_tilt < -2.0°  
-2.0° <= avg_tilt <= 2.0°  
```
### 8.3 MediaPipe Feature Index
face:
```python
forehead_left = landmarks[21]
forehead_right = landmarks[251]

cheek_left = landmarks[234]
cheek_right = landmarks[454]

jaw_left = landmarks[172]
jaw_right = landmarks[397]

forehead_center = landmarks[10]
chin = landmarks[152]
```
eye:
```python
left_outer = landmarks[33]    
left_inner = landmarks[133]   
left_top = landmarks[159]     
left_bottom = landmarks[145]  

right_outer = landmarks[263]
right_inner = landmarks[362]
right_top = landmarks[386]
right_bottom = landmarks[374]
```

---

## 9.Demo Video
Demo video is located at
https://drive.google.com/drive/u/2/my-drive
Demo/Web_AR_Try-On_demo.mp4

---

## 10.Future Work
1. Refine the database to enhance the accuracy of predicting facial shapes and eye shapes
2. Train a genuine decision tree model for classification
3. Develop a recommendation system

---

## 11.Author
Ge Hongrui
hongrui@acestartechsi.com

---

## 12. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
