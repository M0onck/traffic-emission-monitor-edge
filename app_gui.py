import sys
import cv2
import numpy as np

# --- 引入 Qt 相关依赖 ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QTabWidget, QGridLayout, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QPainterPath

# --- 引入业务组件 ---
import infra.config.loader as cfg
from app.monitor_engine import TrafficMonitorEngine
from app.bootstrap import AppBootstrap

# --- UI 组件：双轨物理曲线 (速度 & 加速度) ---
class SpeedCurveWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.speeds = []
        self.accels = []
        self.setMinimumHeight(200) # 稍微加高一点以容纳负半轴
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")

    def update_curve(self, speeds, accels):
        self.speeds = speeds
        self.accels = accels
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 1. 绘制背景
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        
        if len(self.speeds) < 2 or len(self.accels) < 2:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待车辆离场结算数据...")
            return

        # 2. 坐标系与量程设定
        # 统一速度与加速度的最大量程，强制至少为 20.0，且关于 X 轴对称
        max_val = max(20.0, max(self.speeds) * 1.2, max(abs(a) for a in self.accels) * 1.2)
        
        # Y 轴映射函数：将物理值 (-max_val 到 +max_val) 映射到像素 (h 到 0)
        def map_y(val):
            # val / max_val 的范围是 [-1, 1]
            # 屏幕坐标中，0 对应中间，+1 对应顶部(0)，-1 对应底部(h)
            return h / 2 - (val / max_val) * (h / 2)

        # 3. 绘制辅助网格线和零刻度线 (X轴)
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
        for i in range(1, 4):
            if i != 2: # 避开中间的 0 刻度线
                y_line = int(h * i / 4)
                painter.drawLine(0, y_line, w, y_line)
                
        # 绘制纯白色的 0 刻度线 (X轴)
        zero_y = int(h / 2)
        painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))
        painter.drawLine(0, zero_y, w, zero_y)

        # 4. 构建并绘制曲线
        dx = w / max(1, len(self.speeds) - 1)
        path_spd, path_acc = QPainterPath(), QPainterPath()
        
        for i in range(len(self.speeds)):
            x = i * dx
            y_spd = map_y(self.speeds[i])
            y_acc = map_y(self.accels[i])
            
            if i == 0:
                path_spd.moveTo(x, y_spd)
                path_acc.moveTo(x, y_acc)
            else:
                path_spd.lineTo(x, y_spd)
                path_acc.lineTo(x, y_acc)

        # 绘制速度曲线 (青色)
        painter.setPen(QPen(QColor(0, 255, 255), 2))
        painter.drawPath(path_spd)
        
        # 绘制加速度曲线 (红色)
        painter.setPen(QPen(QColor(255, 50, 50), 2))
        painter.drawPath(path_acc)

        # 5. 绘制文字信息标注
        avg_spd = sum(self.speeds) / len(self.speeds)
        max_spd = max(self.speeds)
        
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        
        # 左上角：图例与量程
        painter.setPen(QPen(QColor(0, 255, 255)))
        painter.drawText(10, 25, f"■ 速度 (m/s)")
        painter.setPen(QPen(QColor(255, 50, 50)))
        painter.drawText(10, 45, f"■ 加速度 (m/s²)")
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(10, 70, f"统一量程: ±{max_val:.1f}")
        
        # 右上角：速度统计信息
        painter.setPen(QPen(QColor(255, 255, 255)))
        stats_text = f"平均速度: {avg_spd:.1f} m/s   最大速度: {max_spd:.1f} m/s"
        # 获取文字宽度以靠右对齐
        text_w = painter.fontMetrics().width(stats_text)
        painter.drawText(w - text_w - 15, 25, stats_text)

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
        components = AppBootstrap.setup_components(config)

        # 5. 启动引擎
        # 此时的 components 已经是一个包含了所有依赖的干净字典
        self.engine = TrafficMonitorEngine(config, components, self.emit_frame)
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

        self.phys_w = 20.0 # 默认物理宽度
        self.phys_h = 20.0 # 默认物理长度

        self.is_collecting = False # 标记采集任务是否正在运行
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # QStackedWidget 用于页面切换
        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)

        # 底部导航栏：包裹在一个独立的 Widget 中，方便在主菜单时整体隐藏
        self.nav_widget = QWidget()
        self.nav_layout = QHBoxLayout(self.nav_widget)
        self.nav_layout.setContentsMargins(20, 10, 20, 10)
        
        font = QFont("Arial", 16, QFont.Bold)

        # 返回主页按钮
        self.btn_home = QPushButton("返回主页")
        self.btn_home.setFont(font)
        self.btn_home.setMinimumHeight(50)
        self.btn_home.setStyleSheet("background-color: #f39c12; color: white;") 
        self.btn_home.clicked.connect(self.return_to_home)

        # 结束采集按钮 (默认隐藏)
        self.btn_stop = QPushButton("结束采集")
        self.btn_stop.setFont(font)
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white;") 
        self.btn_stop.clicked.connect(self.stop_collection_trigger)

        self.btn_prev = QPushButton("◀ 上一步")
        self.btn_prev.setFont(font)
        self.btn_prev.setMinimumHeight(50)
        self.btn_prev.clicked.connect(self.prev_page)
        
        self.btn_next = QPushButton("下一步 ▶")
        self.btn_next.setFont(font)
        self.btn_next.setMinimumHeight(50)
        self.btn_next.clicked.connect(self.next_page)

        self.nav_layout.addWidget(self.btn_home)
        self.nav_layout.addWidget(self.btn_stop)
        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.btn_next)
        self.main_layout.addWidget(self.nav_widget)

        # 初始化页面：按顺序压入栈中
        self.init_page_0_main_menu()     # 索引 0：BIOS 主界面
        self.init_page_1_calibration()   # 索引 1：标定步骤
        self.init_page_2_settings()      # 索引 2：设置步骤
        self.init_page_3_monitor()       # 索引 3：运行面板

        self.update_nav_buttons()

    def update_main_menu_btn_style(self):
        """根据采集状态刷新主界面按钮颜色和文字"""
        if self.is_collecting:
            style = """
                QPushButton {
                    background-color: #27ae60; color: white; border: none; border-radius: 8px;
                    padding: 15px; text-align: left; padding-left: 20px;
                }
                QPushButton:hover { background-color: #2ecc71; }
            """
            self.btn_app1.setText("多源数据采集 (运行中...)")
        else:
            style = """
                QPushButton {
                    background-color: #2962ff; color: white; border: none; border-radius: 8px;
                    padding: 15px; text-align: left; padding-left: 20px;
                }
                QPushButton:hover { background-color: #0039cb; }
            """
            self.btn_app1.setText("多源数据采集")
        
        self.btn_app1.setStyleSheet(style)

    def init_page_0_main_menu(self):
        """主调度界面"""
        page = QWidget()
        page.setStyleSheet("background-color: #0f111a;") # 深邃的边缘计算科技蓝/黑底色
        layout = QHBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(30)

        # --- 左侧：硬件与状态信息看板 (BIOS 仪表盘风格) ---
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #1a1d2d; border: 2px solid #2d324f; border-radius: 12px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(25, 25, 25, 25)

        title = QLabel("系统状态")
        title.setFont(QFont("Consolas", 18, QFont.Bold))
        title.setStyleSheet("color: #00e676; border: none;") # 荧光绿点缀
        left_layout.addWidget(title)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border: 1px solid #2d324f;")
        left_layout.addWidget(line)
        left_layout.addSpacing(15)

        # 这里的数值目前是硬编码的占位符，后续可以编写代码去读取树莓派真实的 os/psutil 数据
        import time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        status_info = [
            ("系统时间", current_time),
            ("边缘存储", "112 GB / 256 GB"),
            ("网络连接", "WLAN (192.168.1.10)"),
            ("气象网关", "OFFLINE (端口未接入)"),
            ("CPU 温度", "45.2 °C"),
            ("NPU 温度", "41.0 °C"),
        ]

        font_label = QFont("Consolas", 14)
        font_value = QFont("Consolas", 14, QFont.Bold)
        for key, val in status_info:
            row = QHBoxLayout()
            lbl_k = QLabel(key)
            lbl_k.setStyleSheet("color: #8ab4f8; border: none;")
            lbl_k.setFont(font_label)
            lbl_v = QLabel(val)
            lbl_v.setStyleSheet("color: #ffffff; border: none;")
            lbl_v.setFont(font_value)
            lbl_v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(lbl_k)
            row.addWidget(lbl_v)
            left_layout.addLayout(row)

        left_layout.addStretch()
        layout.addWidget(left_panel, 5) # 左侧宽容度占比 5

        # --- 右侧：应用功能分发列表 ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setAlignment(Qt.AlignTop)

        app_title = QLabel("功能列表")
        app_title.setFont(QFont("Arial", 16, QFont.Bold))
        app_title.setStyleSheet("color: #ffffff;")
        app_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(app_title)
        right_layout.addSpacing(30)

        # 统一的按钮样式模板
        btn_style = """
            QPushButton {
                background-color: #2962ff; color: white; border: none; border-radius: 8px;
                padding: 15px; text-align: left; padding-left: 20px;
            }
            QPushButton:hover { background-color: #0039cb; }
            QPushButton:pressed { background-color: #00227b; }
        """
        
        self.btn_app1 = QPushButton("多源数据采集")
        self.btn_app1.setFont(QFont("Arial", 14, QFont.Bold))
        self.update_main_menu_btn_style() # 初始化样式
        # 点击进入界面（首次进入标定，之后进入看板）
        self.btn_app1.clicked.connect(self.route_app1_click)

        btn_app2 = QPushButton("气象站校准 (开发中)")
        btn_app2.setFont(QFont("Arial", 14, QFont.Bold))
        # 灰色未激活样式
        btn_style2 = btn_style.replace("#2962ff", "#455a64").replace("#0039cb", "#37474f").replace("#00227b", "#263238")
        btn_app2.setStyleSheet(btn_style2)

        btn_app3 = QPushButton("浏览历史数据 (开发中)")
        btn_app3.setFont(QFont("Arial", 14, QFont.Bold))
        # 灰色未激活样式
        btn_app3.setStyleSheet(btn_style2)

        btn_exit = QPushButton("退出程序")
        btn_exit.setFont(QFont("Arial", 14, QFont.Bold))
        # 红色危险操作样式
        btn_style3 = btn_style.replace("#2962ff", "#d50000").replace("#0039cb", "#9b0000").replace("#00227b", "#650000")
        btn_exit.setStyleSheet(btn_style3)
        btn_exit.clicked.connect(self.close)

        right_layout.addWidget(self.btn_app1)
        right_layout.addSpacing(15)
        right_layout.addWidget(btn_app2)
        right_layout.addSpacing(15)
        right_layout.addWidget(btn_app3)
        right_layout.addStretch()
        right_layout.addWidget(btn_exit)

        layout.addWidget(right_panel, 4) # 右侧占比 4

        self.stack.addWidget(page)

    def enter_app(self, target_idx):
        """进入具体功能的槽函数"""
        self.stack.setCurrentIndex(target_idx)
        self.update_nav_buttons()

    def route_app1_click(self):
        """主界面按钮的智能跳转路由"""
        if self.is_collecting:
            # 如果已经在采集中，直接跳过标定和设置，切入监控面板 (Index 3)
            self.enter_app(3)
        else:
            # 如果尚未运行，按照正常流程进入第一步标定环节 (Index 1)
            self.enter_app(1)

    def return_to_home(self):
        """返回主界面"""
        self.stack.setCurrentIndex(0)
        self.update_nav_buttons()
        self.update_main_menu_btn_style()

    def init_page_1_calibration(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("步骤 1/2: 拖拽 4 个角点进行标定")
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
        
        title = QLabel("步骤 2/2: 设置真实物理尺寸")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 全新的粗微调控制器
        def create_adjuster(label_text, init_value, setter_func):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFont(QFont("Arial", 16))
            lbl.setMinimumWidth(220)
            
            # 按钮样式
            btn_style = "QPushButton { font-weight: bold; font-size: 16px; background-color: #333; color: white; border-radius: 5px; } QPushButton:pressed { background-color: #555; }"
            
            btn_minus_coarse = QPushButton("- 1.0")
            btn_minus_fine = QPushButton("- 0.1")
            btn_plus_fine = QPushButton("+ 0.1")
            btn_plus_coarse = QPushButton("+ 1.0")
            
            for btn in [btn_minus_coarse, btn_minus_fine, btn_plus_fine, btn_plus_coarse]:
                btn.setFixedSize(65, 50)
                btn.setStyleSheet(btn_style)
            
            val_lbl = QLabel(f"{init_value:.1f} m")
            val_lbl.setFont(QFont("Arial", 22, QFont.Bold))
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setMinimumWidth(100)
            
            # 闭包状态容器 (使用列表规避 nonlocal 限制)
            state = {'val': init_value}
            
            def make_callback(delta):
                def callback():
                    state['val'] = max(1.0, state['val'] + delta)
                    val_lbl.setText(f"{state['val']:.1f} m")
                    setter_func(state['val'])
                return callback

            btn_minus_coarse.clicked.connect(make_callback(-1.0))
            btn_minus_fine.clicked.connect(make_callback(-0.1))
            btn_plus_fine.clicked.connect(make_callback(0.1))
            btn_plus_coarse.clicked.connect(make_callback(1.0))
            
            row.addStretch()
            row.addWidget(lbl)
            row.addWidget(btn_minus_coarse)
            row.addWidget(btn_minus_fine)
            row.addWidget(val_lbl)
            row.addWidget(btn_plus_fine)
            row.addWidget(btn_plus_coarse)
            row.addStretch()
            return row

        def set_w(val): self.phys_w = val
        def set_h(val): self.phys_h = val

        layout.addLayout(create_adjuster("车道总宽度 (Width): ", self.phys_w, set_w))
        layout.addSpacing(30)
        layout.addLayout(create_adjuster("纵向标定距 (Length): ", self.phys_h, set_h))

        self.stack.addWidget(page)

    def init_page_3_monitor(self):
        self.page3 = QWidget()
        layout = QVBoxLayout(self.page3)
        layout.setContentsMargins(0,0,0,0)
        
        # 创建底部 Tab 栏
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.South) # 页签放在底部
        self.tabs.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(self.tabs)
        
        # --- Tab 1: 实时视频监控 ---
        tab_video = QWidget()
        v_layout = QVBoxLayout(tab_video)
        v_layout.setContentsMargins(0,0,0,0)
        self.video_label = QLabel("正在启动边缘端硬件加速推理，请稍候...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFont(QFont("Arial", 14))
        self.video_label.setStyleSheet("background-color: black; color: white;")
        v_layout.addWidget(self.video_label)
        self.tabs.addTab(tab_video, "目标追踪预览")
        
        # --- Tab 2: 单车抽样 Dashboard ---
        tab_dash = QWidget()
        d_layout = QVBoxLayout(tab_dash)
        d_layout.setContentsMargins(20, 20, 20, 20)
        
        # 全面中文化状态面板
        info_layout = QGridLayout()
        self.lbl_dash_id = QLabel("目标 ID: 等待驶入...")
        self.lbl_dash_type = QLabel("车型: -")
        self.lbl_dash_plate = QLabel("车牌颜色: -")
        self.lbl_dash_dist = QLabel("行驶距离: 0.0 m")
        
        font_dash = QFont("Arial", 14)
        for lbl in [self.lbl_dash_id, self.lbl_dash_type, self.lbl_dash_plate, self.lbl_dash_dist]:
            lbl.setFont(font_dash)
            
        info_layout.addWidget(self.lbl_dash_id, 0, 0)
        info_layout.addWidget(self.lbl_dash_type, 0, 1)
        info_layout.addWidget(self.lbl_dash_plate, 1, 0)
        info_layout.addWidget(self.lbl_dash_dist, 1, 1)
        d_layout.addLayout(info_layout)
        
        d_layout.addSpacing(10)
        
        # 修改标题栏文本
        title_lbl = QLabel("ROI内车辆运动学轨迹:")
        title_lbl.setFont(QFont("Arial", 12, QFont.Bold))
        d_layout.addWidget(title_lbl)
        
        # 曲线组件
        self.curve_widget = SpeedCurveWidget()
        d_layout.addWidget(self.curve_widget)
        d_layout.addStretch()
        
        self.tabs.addTab(tab_dash, "车流捕获数据")
        
        # --- Tab 3: 地表热力瞄准 ---
        tab_thermal = QWidget()
        t_layout = QVBoxLayout(tab_thermal)
        t_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. 顶部温度信息展示布局
        t_info_layout = QHBoxLayout()
        self.lbl_thermal_center = QLabel("中心温度: -- °C")
        self.lbl_thermal_center.setStyleSheet("color: #ff5555; font-weight: bold;") # 中心温度用红色高亮
        self.lbl_thermal_min = QLabel("最低温度: -- °C")
        self.lbl_thermal_max = QLabel("最高温度: -- °C")
        
        font_t = QFont("Arial", 16, QFont.Bold)
        for lbl in [self.lbl_thermal_min, self.lbl_thermal_center, self.lbl_thermal_max]:
            lbl.setFont(font_t)
            lbl.setAlignment(Qt.AlignCenter)
            t_info_layout.addWidget(lbl)
            
        t_layout.addLayout(t_info_layout)
        t_layout.addSpacing(10)
        
        # 2. 热成像画面画布
        self.thermal_label = QLabel("等待热成像传感器接入...")
        self.thermal_label.setAlignment(Qt.AlignCenter)
        self.thermal_label.setStyleSheet("background-color: #111; border: 2px solid #444;")
        # MLX90640 原生是 32x24，我们按比例放大以适应屏幕 (400x300)
        self.thermal_label.setFixedSize(400, 300)
        
        # 使用水平布局让画面居中
        img_layout = QHBoxLayout()
        img_layout.addStretch()
        img_layout.addWidget(self.thermal_label)
        img_layout.addStretch()
        
        t_layout.addLayout(img_layout)
        t_layout.addStretch()
        
        self.tabs.addTab(tab_thermal, "热成像仪数据")

        # 定时器：用于轮询底层数据更新 Dashboard
        self.dash_timer = QTimer(self)
        self.dash_timer.timeout.connect(self.update_timer_tasks)
        self.sampled_tid = None
        
        self.stack.addWidget(self.page3)

    def prev_page(self):
        idx = self.stack.currentIndex()
        if idx > 1: # 防止用户通过“上一步”按钮退回到主菜单
            self.stack.setCurrentIndex(idx - 1)
        self.update_nav_buttons()

    def next_page(self):
        idx = self.stack.currentIndex()
        if idx == 2:
            # 若还未启动则进入运行状态
            if not self.is_collecting:
                self.start_engine()
                self.dash_timer.start(100)  # 10Hz 轮询更新 Dashboard
                self.is_collecting = True
        if idx < self.stack.count() - 1:
            self.stack.setCurrentIndex(idx + 1)
        self.update_nav_buttons()

    def stop_collection_trigger(self):
        """触发结束采集：弹出确认窗口"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("确认操作")
        msg_box.setText("确定要结束当前的采集任务并关闭引擎吗？")
        msg_box.setInformativeText("未保存的缓冲区数据可能会丢失。")
        msg_box.setIcon(QMessageBox.Question)
        
        # 自定义按钮文字
        yes_btn = msg_box.addButton("确定", QMessageBox.YesRole)
        no_btn = msg_box.addButton("取消", QMessageBox.NoRole)
        
        msg_box.exec_()
        
        if msg_box.clickedButton() == yes_btn:
            self.final_stop_process()

    def final_stop_process(self):
        """正式执行退出逻辑"""
        if hasattr(self, 'dash_timer'): self.dash_timer.stop()
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.worker.wait(1000)
        
        self.is_collecting = False
        self.update_main_menu_btn_style()
        self.return_to_home()

    def update_timer_tasks(self):
        """总控定时器：分配 UI 刷新任务"""
        if not hasattr(self, 'worker') or not self.worker.engine: return
        
        self._update_thermal_view()  # 1. 实时刷新热成像
        self._update_dashboard()     # 2. 刷新车辆抽样看板

    def _update_thermal_view(self):
        """处理热力图渲染与数据提取"""
        engine = self.worker.engine
        
        # 检查是否成功挂载了热成像组件
        if not hasattr(engine, 'thermal_cam') or engine.thermal_cam is None:
            return
            
        thermal_matrix = engine.thermal_cam.read()
        if thermal_matrix is not None:

            # 1. 修正画面显示方向
            # 矩阵水平镜像翻转
            thermal_matrix = np.fliplr(thermal_matrix)

            # 2. 提取物理温度 (取中心 2x2 区域的均值更稳定)
            t_min = np.min(thermal_matrix)
            t_max = np.max(thermal_matrix)
            t_center = thermal_matrix[11:13, 15:17].mean() 
            
            self.lbl_thermal_min.setText(f"最低温度: {t_min:.1f} °C")
            self.lbl_thermal_max.setText(f"最高温度: {t_max:.1f} °C")
            self.lbl_thermal_center.setText(f"中心温度: {t_center:.1f} °C")
            
            # 3. 图像视觉渲染 (将浮点温度映射为 RGB 伪彩色)
            # 归一化到 0~255
            norm_img = cv2.normalize(thermal_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 叠加 JET 经典热力图伪彩 (蓝-绿-黄-红)
            color_map = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            
            # 使用双三次插值放大画面，让原本粗糙的 32x24 变得平滑柔和
            display_w, display_h = 400, 300
            display_img = cv2.resize(color_map, (display_w, display_h), interpolation=cv2.INTER_CUBIC)
            
            # 4. 绘制中心十字准星 (Crosshair)
            cx, cy = display_w // 2, display_h // 2
            color_cross = (255, 255, 255) # 白色准星
            # 画十字
            cv2.line(display_img, (cx - 20, cy), (cx - 5, cy), color_cross, 2)
            cv2.line(display_img, (cx + 5, cy), (cx + 20, cy), color_cross, 2)
            cv2.line(display_img, (cx, cy - 20), (cx, cy - 5), color_cross, 2)
            cv2.line(display_img, (cx, cy + 5), (cx, cy + 20), color_cross, 2)
            # 中心打个小点
            cv2.circle(display_img, (cx, cy), 1, color_cross, -1)
            
            # 4. 转交 Qt 显示
            rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            qimg = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
            self.thermal_label.setPixmap(QPixmap.fromImage(qimg))

    def _update_dashboard(self):
        """Dashboard 数据抽样更新逻辑 (改为离场后结算展示)"""
        # 确保后台引擎已经初始化
        if not hasattr(self, 'worker') or not self.worker.engine: return
        engine = self.worker.engine
        
        # 检查引擎是否产出了最新结算完毕的离场数据
        if not hasattr(engine, 'latest_exit_record') or not engine.latest_exit_record:
            return
            
        latest_data = engine.latest_exit_record
        tid = latest_data['tid']

        # 如果这辆车已经在 Dashboard 上展示过了，就不重复刷新，避免闪烁
        if getattr(self, 'sampled_tid', None) == tid:
            return
            
        # 捕获到了新的离场车辆！
        self.sampled_tid = tid
        record = latest_data['record']
        type_str = latest_data['type_str']
        
        # 读取引擎层结算好的投票结果，保持与落盘数据 100% 一致
        plate_color = record.get('final_plate_color', 'Unknown')
            
        # 提取经过 S-G 非因果滤波处理过的高质量速度、加速度曲线
        trajectory = record.get('trajectory', [])
        speeds = [p['speed'] for p in trajectory if 'speed' in p]
        accels = [p['accel'] for p in trajectory if 'accel' in p]
        
        # 更新 UI 组件
        self.lbl_dash_id.setText(f"目标 ID: #{tid} (已离场结算)")
        display_type = type_str if type_str else "未知"
        self.lbl_dash_type.setText(f"车型: {display_type}")
        color_map = {'blue': '蓝色', 'green': '绿色', 'yellow': '黄色', 'white': '白色', 'black': '黑色', 'Unknown': '未知'}
        zh_color = color_map.get(plate_color, plate_color)
        self.lbl_dash_plate.setText(f"车牌颜色: {zh_color}")
        self.lbl_dash_dist.setText(f"行驶里程: {record.get('total_distance_m', 0.0):.1f} m")
        
        self.curve_widget.update_curve(speeds, accels)

    def update_nav_buttons(self):
        idx = self.stack.currentIndex()
        
        # 核心逻辑：如果在 BIOS 界面 (0)，直接隐藏底部的所有控制按钮
        self.nav_widget.setVisible(idx > 0)
        if idx == 0:
            return
        
        # 只有在运行面板（Index 3）且正在采集时，才显示“结束采集”按钮
        self.btn_stop.setVisible(idx == 3 and self.is_collecting)

        # 现在的步骤页面索引是 1 -> 2 -> 3
        self.btn_prev.setVisible(idx > 1 and idx < 3) 
        
        if idx == 1:
            self.btn_next.setVisible(True)
            self.btn_next.setText("下一步 ▶")
            self.btn_next.setStyleSheet("")
        elif idx == 2:
            self.btn_next.setVisible(True)
            self.btn_next.setText(" 开 始 ")
            self.btn_next.setStyleSheet("background-color: #4CAF50; color: white;")
        elif idx == 3:
            self.btn_next.setVisible(False) 
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
        if hasattr(self, 'dash_timer'):
            self.dash_timer.stop() # 安全停止定时器
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.worker.wait(2000) # 等待线程安全退出
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficMonitorUI()
    window.showFullScreen()
    sys.exit(app.exec_())
