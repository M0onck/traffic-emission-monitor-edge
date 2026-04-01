# perception/sensor/thermal_camera.py
from ctypes import *
import numpy as np
import threading
import time
import os

class ThermalCamera:
    """
    [感知层] 热成像传感器异步驱动
    负责在后台线程安全地调用 .so 库读取 MLX90640 数据，防止阻塞主视频流。
    """
    def __init__(self, lib_path='./libmlx90640.so'):
        if not os.path.exists(lib_path):
            print(f">>> [Warning] 热成像动态库未找到: {lib_path}")
            self.lib = None
            return
            
        self.lib = cdll.LoadLibrary(lib_path)
        
        # 预先分配 C 语言的 float 数组内存 (32 * 24 = 768)
        self._temp_array = (c_float * 768)()
        self._ptemp = pointer(self._temp_array)
        
        self.latest_frame = None
        self._running = False
        self._lock = threading.Lock()
        
    def start(self):
        if self.lib is None: return
        print(">>> [ThermalCamera] 启动热成像后台采集线程...")
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        while self._running:
            try:
                # 调用底层 C 库读取数据 (这是阻塞型操作)
                self.lib.get_mlx90640_temp(self._ptemp)
                
                # 将 C 数组转化为 Numpy 数组，并 reshape 为图像矩阵 (高度24, 宽度32)
                frame = np.array(self._temp_array, dtype=np.float32).reshape((24, 32))
                
                # 加锁更新最新帧，供主线程读取
                with self._lock:
                    self.latest_frame = frame
                    
            except Exception as e:
                print(f"[ThermalCamera] 读取异常: {e}")
                time.sleep(0.1)

    def read(self) -> np.ndarray:
        """非阻塞读取最新的一帧热成像数据 (24x32 的二维温度矩阵)"""
        if self.lib is None: return None
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=1.0)
            print(">>> [ThermalCamera] 采集线程已安全停止。")
