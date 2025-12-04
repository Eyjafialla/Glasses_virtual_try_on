# ===== camera_async.py =====
# 异步摄像头读取 - 使用独立线程预读帧
# 可将摄像头读取时间从63ms降低到10ms以下

import cv2
import threading
import time


class AsyncVideoCapture:
    """
    异步视频捕获 - 在后台线程中持续读取最新帧
    
    优势:
    - 主线程read()立即返回最新帧，不等待摄像头IO
    - 可将读取时间从60ms降低到1-2ms
    """
    
    def __init__(self, src=0, width=None, height=None, fps=None):
        """
        Args:
            src: 摄像头索引
            width: 捕获宽度
            height: 捕获高度  
            fps: 目标帧率
        """
        self.cap = cv2.VideoCapture(src)
        
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 读取第一帧
        self.ret, self.frame = self.cap.read()
        
        # 线程控制
        self.lock = threading.Lock()
        self.running = True
        
        # 启动后台读取线程
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
        # 性能统计
        self.read_count = 0
        self.last_read_time = time.time()
    
    def _reader(self):
        """后台线程 - 持续读取最新帧"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    self.read_count += 1
            else:
                # 如果读取失败，稍微等待后重试
                time.sleep(0.01)
    
    def read(self):
        """
        主线程调用 - 立即返回最新帧
        
        Returns:
            ret, frame: 和cv2.VideoCapture.read()相同的返回值
        """
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
    
    def get(self, prop_id):
        """获取摄像头属性"""
        return self.cap.get(prop_id)
    
    def set(self, prop_id, value):
        """设置摄像头属性"""
        return self.cap.set(prop_id, value)
    
    def release(self):
        """释放资源"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
    
    def get_read_fps(self):
        """获取后台线程的实际读取帧率"""
        now = time.time()
        dt = now - self.last_read_time
        if dt > 0:
            fps = self.read_count / dt
            self.read_count = 0
            self.last_read_time = now
            return fps
        return 0
    
    def __del__(self):
        self.release()


# 使用示例
if __name__ == "__main__":
    import time
    
    # 对比测试
    print("=== 测试同步 vs 异步摄像头读取 ===\n")
    
    # 同步读取测试
    print("[1] 同步读取测试:")
    cap_sync = cv2.VideoCapture(1)
    cap_sync.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_sync.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    times_sync = []
    for i in range(30):
        t0 = time.perf_counter()
        ret, frame = cap_sync.read()
        t1 = time.perf_counter()
        times_sync.append((t1 - t0) * 1000)
    
    avg_sync = sum(times_sync) / len(times_sync)
    print(f"  平均读取时间: {avg_sync:.2f}ms")
    cap_sync.release()
    
    # 异步读取测试
    print("\n[2] 异步读取测试:")
    cap_async = AsyncVideoCapture(src=1, width=1280, height=720)
    time.sleep(0.1)  # 等待后台线程启动
    
    times_async = []
    for i in range(30):
        t0 = time.perf_counter()
        ret, frame = cap_async.read()
        t1 = time.perf_counter()
        times_async.append((t1 - t0) * 1000)
    
    avg_async = sum(times_async) / len(times_async)
    print(f"  平均读取时间: {avg_async:.2f}ms")
    print(f"  后台读取FPS: {cap_async.get_read_fps():.1f}")
    cap_async.release()
    
    # 结果对比
    print(f"\n[结果] 异步读取比同步快 {avg_sync/avg_async:.1f}x")
    print(f"  预期节省时间: {avg_sync - avg_async:.2f}ms/帧")
