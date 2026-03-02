import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

# --- 引入原有业务组件 ---
import infra.config.loader as cfg
from app.monitor_engine import TrafficMonitorEngine
from domain.vehicle.repository import VehicleRegistry
from domain.physics.vsp_calculator import VSPCalculator
from domain.physics.opmode_calculator import MovesOpModeCalculator
from domain.physics.brake_emission_model import BrakeEmissionModel
from domain.physics.tire_emission_model import TireEmissionModel
from domain.vehicle.classifier import VehicleClassifier
from infra.store.sqlite_manager import DatabaseManager
from infra.sys.process_optimizer import SystemOptimizer
from perception.math.geometry import ViewTransformer
from perception.kinematics_estimator import KinematicsEstimator
from ui.renderer import Visualizer
from ui.console_reporter import Reporter
from perception.gst_pipeline import GstPipelineManager
import supervision as sv

# --- UI 组件：可拖拽的标定画布 ---
class CalibrationCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.orig_frame = None
        self.real_points = []
        self.drag_idx = -1
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 380)

    def load_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.orig_frame = frame
            self.init_points()
            self.update_display()

    def init_points(self):
        h, w = self.orig_frame.shape[:2]
        cx, cy = w // 2, h // 2
        dx, dy = int(w * 0.2), int(h * 0.2)
        # 核心修复：将点初始化在“原始视频真实分辨率”的坐标系下
        self.real_points = [[cx - dx, cy + dy], [cx + dx, cy + dy], 
                            [cx + dx, cy - dy], [cx - dx, cy - dy]]

    def _get_transform_params(self):
        """计算画布到真实分辨率的双向映射参数"""
        cw, ch = self.width(), self.height()
        if cw < 100 or ch < 100: # 窗口尚未完全渲染时的默认回退尺寸
            cw, ch = 800, 380
            
        oh, ow = self.orig_frame.shape[:2]
        scale = min(cw / ow, ch / oh)
        nw, nh = int(ow * scale), int(oh * scale)
        x_off = (cw - nw) // 2
        y_off = (ch - nh) // 2
        return scale, x_off, y_off, nw, nh, cw, ch

    def update_display(self):
        if self.orig_frame is None: return
        
        scale, x_off, y_off, nw, nh, cw, ch = self._get_transform_params()
        
        # 构建带黑边的 UI 画布 (与第三步运行时的 resize_with_pad 逻辑一致)
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        resized = cv2.resize(self.orig_frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
        
        # 正向映射：真实坐标 -> UI坐标
        pts_ui = []
        for rx, ry in self.real_points:
            ux = int(rx * scale + x_off)
            uy = int(ry * scale + y_off)
            pts_ui.append([ux, uy])
            
        pts_ui_arr = np.array(pts_ui, np.int32)
        cv2.polylines(canvas, [pts_ui_arr], True, (0, 255, 0), 2)
        
        labels = ["BL", "BR", "TR", "TL"]
        for i, (ux, uy) in enumerate(pts_ui):
            color = (0, 0, 255) if i == self.drag_idx else (0, 255, 0)
            cv2.circle(canvas, (ux, uy), 15, color, -1)
            cv2.putText(canvas, labels[i], (ux+20, uy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        rgb_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch_ = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, ch_ * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, event):
        if self.orig_frame is None: return
        x, y = event.x(), event.y()
        scale, x_off, y_off, _, _, _, _ = self._get_transform_params()
        
        hit_radius = 40 
        for i, (rx, ry) in enumerate(self.real_points):
            # 将真实坐标映射到 UI 坐标以响应点击事件
            ux = rx * scale + x_off
            uy = ry * scale + y_off
            if np.hypot(x - ux, y - uy) < hit_radius:
                self.drag_idx = i
                self.update_display()
                break

    def mouseMoveEvent(self, event):
        if self.drag_idx != -1 and self.orig_frame is not None:
            x, y = event.x(), event.y()
            scale, x_off, y_off, _, _, cw, ch = self._get_transform_params()
            
            # 反向映射：UI拖拽坐标 -> 真实视频坐标
            rx = (x - x_off) / scale
            ry = (y - y_off) / scale
            
            # 限制标定点不要拖出原生图像的物理范围
            oh, ow = self.orig_frame.shape[:2]
            rx = max(0, min(ow, rx))
            ry = max(0, min(oh, ry))
            
            self.real_points[self.drag_idx] = [rx, ry]
            self.update_display()

    def mouseReleaseEvent(self, event):
        self.drag_idx = -1
        self.update_display()
        
    def resizeEvent(self, event):
        """监听窗口尺寸变化"""
        super().resizeEvent(event)
        # 一旦布局管理器分配了最终的画布尺寸，立即重绘带黑边的等比画面
        if self.orig_frame is not None:
            self.update_display()

    def get_real_points(self):
        """向后台提供 0.0 ~ 1.0 的归一化坐标，彻底解决分辨率不一致导致的偏移"""
        pts = np.array(self.real_points, dtype=np.float32)
        oh, ow = self.orig_frame.shape[:2]
        pts[:, 0] /= ow  # 转换为 0~1 的比例值
        pts[:, 1] /= oh
        return pts

# --- 业务线程：将 Hailo 推理放到后台 ---
class EngineWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, source_points, phys_w, phys_h):
        super().__init__()
        self.source_points = source_points
        self.phys_w = phys_w
        self.phys_h = phys_h
        self.engine = None

    def run(self):
        # 此处复用原 bootstrap.py 的初始化逻辑
        print(">>> [System] 初始化后台推理引擎...")
        
        target_points = np.array([
            [0, 0], [self.phys_w, 0], 
            [self.phys_w, self.phys_h], [0, self.phys_h]
        ], dtype=np.float32)

        # 初始占位符，由 Engine 获取第一帧时动态还原真实分辨率坐标
        dummy_pts = np.zeros((4, 2), dtype=np.float32)

        opmode_calculator = MovesOpModeCalculator(config=cfg._e)
        gst_config = {"video_path": cfg.VIDEO_PATH}
        camera_manager = GstPipelineManager(gst_config)

        components = {
            'camera': camera_manager,
            'norm_source_points': self.source_points, # 传入刚才获取的归一化坐标
            'target_points': target_points,           # 传入目标物理坐标系
            'transformer': ViewTransformer(dummy_pts, target_points),
            'visualizer': Visualizer(
                calibration_points=dummy_pts,
                trace_length=cfg.FPS,
                opmode_calculator=opmode_calculator
            ),
            'registry': VehicleRegistry(fps=cfg.FPS, min_survival_frames=cfg.MIN_SURVIVAL_FRAMES),
            'db': DatabaseManager(cfg.DB_PATH, cfg.FPS),
            'classifier': VehicleClassifier(cfg.TYPE_MAP, {"car": cfg.YOLO_CLASS_CAR, "bus": cfg.YOLO_CLASS_BUS, "truck": cfg.YOLO_CLASS_TRUCK})
        }

        # 根据配置加载其他组件 (简化展示，可自行补全完整参数)
        if cfg.ENABLE_MOTION:
            components['kinematics'] = KinematicsEstimator({"fps": cfg.FPS, "kinematics": {"speed_window": 15, "accel_window": 15}})
        
        if getattr(cfg, "ENABLE_OCR", False):
            from infra.concurrency.plate_worker import PlateClassifierWorker
            pw = PlateClassifierWorker(cfg.Y5FU_PATH, cfg.LITEMODEL_PATH)
            pw.start()
            components['plate_worker'] = pw

        self.engine = TrafficMonitorEngine(cfg, components, frame_callback=self.emit_frame)
        self.engine.run()

    def emit_frame(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_ready.emit(rgb_img)
        
    def stop(self):
        if self.engine:
            self.engine._is_running = False

# --- 主窗口：管理三个步骤 ---
class TrafficMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Emission Edge Monitor")

        # 隐藏操作系统的窗口边框和标题栏
        self.setWindowFlags(Qt.FramelessWindowHint) 
        
        # 强制设置为树莓派屏幕大小
        self.setFixedSize(800, 480)

        self.phys_w = 10.5 # 默认物理宽度
        self.phys_h = 30.0 # 默认物理长度

        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # QStackedWidget 用于页面切换
        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)

        # 底部导航栏
        self.nav_layout = QHBoxLayout()
        self.nav_layout.setContentsMargins(20, 10, 20, 10)
        
        font = QFont("Arial", 16, QFont.Bold)

        # 退出按钮
        self.btn_exit = QPushButton("退出")
        self.btn_exit.setFont(font)
        self.btn_exit.setMinimumHeight(50)
        self.btn_exit.setStyleSheet("background-color: #f44336; color: white;") # 红色警告色
        self.btn_exit.clicked.connect(self.close) # 点击直接调用窗口关闭事件

        self.btn_prev = QPushButton("◀ 上一步")
        self.btn_prev.setFont(font)
        self.btn_prev.setMinimumHeight(50)
        self.btn_prev.clicked.connect(self.prev_page)
        
        self.btn_next = QPushButton("下一步 ▶")
        self.btn_next.setFont(font)
        self.btn_next.setMinimumHeight(50)
        self.btn_next.clicked.connect(self.next_page)

        self.nav_layout.addWidget(self.btn_exit)
        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.btn_next)
        self.main_layout.addLayout(self.nav_layout)

        # 初始化页面
        self.init_page_1_calibration()
        self.init_page_2_settings()
        self.init_page_3_monitor()

        self.update_nav_buttons()

    def init_page_1_calibration(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("步骤 1/3: 触屏拖拽 4 个角点进行标定")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        self.canvas = CalibrationCanvas()
        self.canvas.load_frame(cfg.VIDEO_PATH) # 加载第一帧
        layout.addWidget(self.canvas)
        
        self.stack.addWidget(page)

    def init_page_2_settings(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("步骤 2/3: 设置真实物理尺寸")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 尺寸调节器闭包生成器
        def create_adjuster(label_text, value, step, setter_func):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFont(QFont("Arial", 16))
            
            btn_minus = QPushButton(" - ")
            btn_minus.setFont(QFont("Arial", 20, QFont.Bold))
            btn_minus.setFixedSize(60, 60)
            
            val_lbl = QLabel(f"{value:.1f} m")
            val_lbl.setFont(QFont("Arial", 20, QFont.Bold))
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setMinimumWidth(120)
            
            btn_plus = QPushButton(" + ")
            btn_plus.setFont(QFont("Arial", 20, QFont.Bold))
            btn_plus.setFixedSize(60, 60)
            
            def on_minus():
                nonlocal value
                value = max(1.0, value - step)
                val_lbl.setText(f"{value:.1f} m")
                setter_func(value)
                
            def on_plus():
                nonlocal value
                value += step
                val_lbl.setText(f"{value:.1f} m")
                setter_func(value)
                
            btn_minus.clicked.connect(on_minus)
            btn_plus.clicked.connect(on_plus)
            
            row.addStretch()
            row.addWidget(lbl)
            row.addWidget(btn_minus)
            row.addWidget(val_lbl)
            row.addWidget(btn_plus)
            row.addStretch()
            return row

        def set_w(val): self.phys_w = val
        def set_h(val): self.phys_h = val

        layout.addLayout(create_adjuster("车道宽度 (Width): ", self.phys_w, 0.5, set_w))
        layout.addSpacing(20)
        layout.addLayout(create_adjuster("分析距离 (Length): ", self.phys_h, 1.0, set_h))

        self.stack.addWidget(page)

    def init_page_3_monitor(self):
        self.page3 = QWidget()
        layout = QVBoxLayout(self.page3)
        layout.setContentsMargins(0,0,0,0)
        self.video_label = QLabel("正在启动边缘端硬件加速推理，请稍候...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFont(QFont("Arial", 14))
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label)
        
        self.stack.addWidget(self.page3)

    def prev_page(self):
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
        self.update_nav_buttons()

    def next_page(self):
        idx = self.stack.currentIndex()
        if idx == 1:
            # 即将进入运行状态
            self.start_engine()
        if idx < self.stack.count() - 1:
            self.stack.setCurrentIndex(idx + 1)
        self.update_nav_buttons()

    def update_nav_buttons(self):
        idx = self.stack.currentIndex()
        self.btn_prev.setVisible(idx > 0 and idx < 2) # 运行中隐藏上一步
        
        if idx == 0:
            self.btn_next.setText("下一步 ▶")
            self.btn_next.setStyleSheet("")
        elif idx == 1:
            self.btn_next.setText(" 🚀 开 始 ")
            self.btn_next.setStyleSheet("background-color: #4CAF50; color: white;")
        elif idx == 2:
            self.btn_next.setVisible(False) # 运行中隐藏下一步按钮
            self.btn_prev.setVisible(False)
            
    def start_engine(self):
        # 提取的是反算好的原生 1080p 真实坐标
        source_points = self.canvas.get_real_points()
        
        # 启动后台引擎线程
        self.worker = EngineWorker(source_points, self.phys_w, self.phys_h)
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.start()

    def update_video_frame(self, rgb_img):
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 利用 Qt 的机制自适应当前 Label 尺寸，自动补齐黑边并保持等比缩放
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        print(">>> [System] 正在安全退出...")
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.worker.wait(2000) # 等待线程安全退出
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficMonitorUI()
    window.showFullScreen()
    sys.exit(app.exec_())
