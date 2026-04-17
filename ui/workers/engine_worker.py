import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import infra.config.loader as cfg
from app.monitor_engine import TrafficMonitorEngine
from app.bootstrap import AppBootstrap

# --- 业务线程：将 Hailo 推理放到后台 ---
class EngineWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, source_points, phys_w, phys_h, weather_station=None):
        super().__init__()
        self.source_points = source_points
        self.phys_w = phys_w
        self.phys_h = phys_h
        self.weather_station = weather_station
        self.engine = None

    def set_prebuilt_components(self, components):
        """接收从外部（Controller）注入的已实例化组件，如摄像头"""
        self.prebuilt_components = components

    def run(self):
        # 1. 加载基础配置
        config = cfg
        
        # 2. 注入来自 UI 交互的动态参数
        # 将用户在界面上拉好的坐标点转换后存入 config，供引导程序使用
        config.SOURCE_POINTS = self.source_points.tolist()
        config.PHYS_WIDTH = self.phys_w
        config.PHYS_HEIGHT = self.phys_h

        # 3. 根据 UI 面板的宽(W)和高(H)，严格构建物理坐标系的四个角点
        # 顺序必须对应图像点击的四个角：[左下(BL), 右下(BR), 右上(TR), 左上(TL)]
        config.TARGET_POINTS = [
            [0, self.phys_h],             # BL: x=0, y=高
            [self.phys_w, self.phys_h],   # BR: x=宽, y=高
            [self.phys_w, 0],             # TR: x=宽, y=0
            [0, 0]                        # TL: x=0, y=0
        ]

        # 4. 装配所有组件
        # 引导模块会根据 config 自动创建 db, registry, camera, plate_worker 等
        if hasattr(self, 'prebuilt_components') and self.prebuilt_components:
            components = self.prebuilt_components
            base_comps = AppBootstrap.setup_components(config)
            for k, v in base_comps.items():
                if k not in components:
                    components[k] = v
        else:
            components = AppBootstrap.setup_components(config)

        if getattr(self, 'weather_station', None):
            components['weather_station'] = self.weather_station

        # 5. 启动引擎
        self.engine = TrafficMonitorEngine(config, components, self.emit_frame)
        self.engine.run()

    def emit_frame(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_ready.emit(rgb_img)
        
    def stop(self):
        if self.engine:
            self.engine._is_running = False
