# ui/workers/engine_worker.py
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class EngineWorker(QThread):
    # 定义发往 GUI 线程的 RGB 图像信号
    frame_ready = pyqtSignal(np.ndarray)

    # 定义发往 GUI 的进度信号 (百分比, 提示文本)
    stop_progress = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self.engine = None
        self._can_emit = True # 帧锁，防止 UI 渲染积压导致卡顿

    def set_engine(self, engine):
        """注入控制器已经装配好的引擎实例"""
        self.engine = engine

    def run(self):
        """后台线程仅负责启动引擎循环"""
        self.engine.shutdown_progress_callback = lambda val, msg: self.stop_progress.emit(val, msg)
        if self.engine:
            self.engine.run()

    def emit_frame(self, frame):
        """作为 MonitorEngine 的回调函数，负责流控和格式转换"""
        if self._can_emit:
            self._can_emit = False # 上锁
            # 在后台线程完成耗时的格式转换
            import cv2
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(rgb_img)

    def unlock_frame(self):
        """由 UI 线程在渲染完成后调用，允许发送下一帧"""
        self._can_emit = True
