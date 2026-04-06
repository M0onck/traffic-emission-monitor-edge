from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QStackedWidget, QTabWidget, 
                             QGridLayout, QFrame, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QComboBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap

import infra.config.loader as cfg
from ui.components.speed_curve import SpeedCurveWidget
from ui.components.calibration_canvas import CalibrationCanvas

class MainWindow(QMainWindow):
    """纯粹的 View 层：只负责界面布局，不处理业务逻辑"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Emission Edge Monitor")
        self.setWindowFlags(Qt.FramelessWindowHint) 
        self.setFixedSize(800, 480)

        # 视图层持有物理尺寸的默认状态
        self.phys_w = 20 # 默认物理宽度
        self.phys_h = 20 # 默认物理长度

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

        # 结束采集按钮 (默认隐藏)
        self.btn_stop = QPushButton("结束采集")
        self.btn_stop.setFont(font)
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white;") 

        # 批量删除按钮 (红色预警，默认隐藏)
        self.btn_delete_db = QPushButton("批量删除")
        self.btn_delete_db.setFont(font)
        self.btn_delete_db.setMinimumHeight(50)
        self.btn_delete_db.setStyleSheet("background-color: #d50000; color: white; border-radius: 5px;")
        self.btn_delete_db.setVisible(False)

        self.btn_prev = QPushButton("◀ 上一步")
        self.btn_prev.setFont(font)
        self.btn_prev.setMinimumHeight(50)
        
        self.btn_next = QPushButton("下一步 ▶")
        self.btn_next.setFont(font)
        self.btn_next.setMinimumHeight(50)

        self.nav_layout.addWidget(self.btn_home)
        self.nav_layout.addWidget(self.btn_stop)
        self.nav_layout.addWidget(self.btn_delete_db)
        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.btn_next)
        self.main_layout.addWidget(self.nav_widget)

        # 初始化页面：按顺序压入栈中
        self.init_page_0_main_menu()     # 索引 0：主界面
        self.init_page_1_calibration()   # 索引 1：标定步骤
        self.init_page_2_settings()      # 索引 2：设置步骤
        self.init_page_3_monitor()       # 索引 3：运行面板
        self.init_page_4_weather_calib() # 索引 4：气象站校准页面
        self.init_page_5_db_browser()    # 索引 5：数据库浏览页面

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

        # 定义一个字典作为类成员，用于保存所有数值 Label 的句柄
        self.status_labels = {}
        
        # 定义需要展示的监控项（初始值先用占位符 "--"，稍后由 Controller 填写真实数据）
        status_keys = [
            "系统时间",
            "边缘存储",
            "网络连接",
            "气象网关",
            "CPU 温度",
            "NPU 温度"
        ]

        font_label = QFont("Consolas", 14)
        font_value = QFont("Consolas", 14, QFont.Bold)
        
        # 循环创建 UI 组件，并将值 Label 存入字典
        for key in status_keys:
            row = QHBoxLayout()
            
            lbl_k = QLabel(key)
            lbl_k.setStyleSheet("color: #8ab4f8; border: none;")
            lbl_k.setFont(font_label)
            
            lbl_v = QLabel("--")  # 默认占位符
            lbl_v.setStyleSheet("color: #ffffff; border: none;")
            lbl_v.setFont(font_value)
            lbl_v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # 将对应项的 QLabel 存入字典
            self.status_labels[key] = lbl_v
            
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
        self.btn_app1.setStyleSheet(btn_style)

        self.btn_app2 = QPushButton("气象设备校准")
        self.btn_app2.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_app2.setStyleSheet(btn_style)

        self.btn_app3 = QPushButton("浏览历史数据")
        self.btn_app3.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_app3.setStyleSheet(btn_style)

        self.btn_exit = QPushButton("退出程序")
        self.btn_exit.setFont(QFont("Arial", 14, QFont.Bold))
        # 红色危险操作样式
        btn_style3 = btn_style.replace("#2962ff", "#d50000").replace("#0039cb", "#9b0000").replace("#00227b", "#650000")
        self.btn_exit.setStyleSheet(btn_style3)
        self.btn_exit.clicked.connect(self.close)

        right_layout.addWidget(self.btn_app1)
        right_layout.addSpacing(15)
        right_layout.addWidget(self.btn_app2)
        right_layout.addSpacing(15)
        right_layout.addWidget(self.btn_app3)
        right_layout.addStretch()
        right_layout.addWidget(self.btn_exit)

        layout.addWidget(right_panel, 4) # 右侧占比 4

        self.stack.addWidget(page)

    def init_page_1_calibration(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        self.lbl_calib_title = QLabel("步骤 1/2: 拖拽 4 个角点进行标定")
        self.lbl_calib_title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(self.lbl_calib_title)
        
        self.canvas = CalibrationCanvas()
        self.canvas.load_frame(cfg.VIDEO_PATH) # 加载第一帧
        layout.addWidget(self.canvas)
        
        self.stack.addWidget(page)
    
    def init_page_2_settings(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        
        self.lbl_settings_title = QLabel("步骤 2/2: 设置真实物理尺寸")
        self.lbl_settings_title.setFont(QFont("Arial", 18, QFont.Bold))
        self.lbl_settings_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_settings_title)

        # 粗微调控制器
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

        # --- Tab 4: 气象数据监测 ---
        tab_weather = QWidget()
        w_layout = QHBoxLayout(tab_weather)
        w_layout.setContentsMargins(20, 20, 20, 20)
        w_layout.setSpacing(20)
        
        # === 左侧：气象数据看板 (3行2列) ===
        weather_left_panel = QFrame()
        weather_left_panel.setStyleSheet("background-color: #1a1d2d; border: 2px solid #2d324f; border-radius: 12px;")
        w_grid_layout = QGridLayout(weather_left_panel)
        w_grid_layout.setContentsMargins(25, 25, 25, 25)
        w_grid_layout.setSpacing(20)
        
        # 重命名为 monitor_labels，与校准页区分
        self.weather_monitor_labels = {}
        
        weather_items = [
            ("温度", "°C"), ("湿度", "%"),
            ("风速", "m/s"), ("风向", "°"),
            ("PM2.5", "μg/m³"), ("PM10", "μg/m³")
        ]
        
        font_w_title = QFont("Arial", 14)
        font_w_val = QFont("Consolas", 20, QFont.Bold)
        
        for i, (name, unit) in enumerate(weather_items):
            row = i // 2
            col = i % 2
            cell_widget = QWidget()
            cell_layout = QVBoxLayout(cell_widget)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl_title = QLabel(name)
            lbl_title.setFont(font_w_title)
            lbl_title.setStyleSheet("color: #8ab4f8; border: none;")
            
            lbl_val = QLabel(f"-- {unit}")
            lbl_val.setFont(font_w_val)
            lbl_val.setStyleSheet("color: #00e676; border: none;")
            lbl_val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            self.weather_monitor_labels[name] = lbl_val 
            
            cell_layout.addWidget(lbl_title)
            cell_layout.addWidget(lbl_val)
            cell_layout.addStretch()
            w_grid_layout.addWidget(cell_widget, row, col)
            
        w_layout.addWidget(weather_left_panel, 7)
        
        # === 右侧：气象站插画展示区 ===
        weather_right_panel = QFrame()
        # 修改边框样式为实线，保持视觉统一
        weather_right_panel.setStyleSheet("background-color: #0f111a; border: 2px solid #2d324f; border-radius: 12px;")
        w_right_layout = QVBoxLayout(weather_right_panel)
        w_right_layout.setContentsMargins(10, 10, 10, 10)
        
        self.lbl_weather_img = QLabel()
        self.lbl_weather_img.setAlignment(Qt.AlignCenter)
        
        # 加载 PNG 图片
        pixmap = QPixmap("resources/weather_station.png")
        
        if not pixmap.isNull():
            # 1. 精确物理可用尺寸 (依据 800x480 分辨率及当前 Margin 倒推)
            target_w, target_h = 200, 310
            
            # 2. 核心要求 A：保持原图比例，锁定高度缩放
            scaled_pixmap = pixmap.scaledToHeight(target_h, Qt.SmoothTransformation)
            
            # 3. 核心要求 B：如果水平方向太长，则从中间截断
            if scaled_pixmap.width() > target_w:
                # 计算中心裁剪的起始 X 坐标
                crop_x = (scaled_pixmap.width() - target_w) // 2
                # 执行裁剪 copy(x, y, width, height)
                final_pixmap = scaled_pixmap.copy(crop_x, 0, target_w, target_h)
            else:
                final_pixmap = scaled_pixmap
                
            self.lbl_weather_img.setPixmap(final_pixmap)
            self.lbl_weather_img.setScaledContents(False) # 必须关闭自动拉伸，否则会破坏原图比例
        else:
            self.lbl_weather_img.setText("图像加载失败\n请检查 resources 目录")
            self.lbl_weather_img.setStyleSheet("color: #e74c3c; border: none;")
            self.lbl_weather_img.setFont(QFont("Arial", 12))
            
        w_right_layout.addWidget(self.lbl_weather_img)
        
        w_layout.addWidget(weather_right_panel, 3) # 保持右侧占比 3
        self.tabs.addTab(tab_weather, "气象数据监测")
        
        self.stack.addWidget(self.page3)
    
    def init_page_4_weather_calib(self):
        """气象站独立校准面板 (Index 4)"""
        page = QWidget()
        page.setStyleSheet("background-color: #0f111a;")
        layout = QHBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(30)
        
        # === 左侧：气象数据看板 + 时间戳 ===
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #1a1d2d; border: 2px solid #2d324f; border-radius: 12px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(25, 25, 25, 25)
        
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(20)
        
        self.weather_calib_labels = {}
        weather_items = [
            ("温度", "°C"), ("湿度", "%"),
            ("风速", "m/s"), ("风向", "°"),
            ("PM2.5", "μg/m³"), ("PM10", "μg/m³")
        ]
        font_w_title = QFont("Arial", 14)
        font_w_val = QFont("Consolas", 20, QFont.Bold)
        
        for i, (name, unit) in enumerate(weather_items):
            row = i // 2; col = i % 2
            cell_widget = QWidget()
            cell_layout = QVBoxLayout(cell_widget)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl_title = QLabel(name)
            lbl_title.setFont(font_w_title)
            lbl_title.setStyleSheet("color: #8ab4f8; border: none;")
            lbl_val = QLabel(f"-- {unit}")
            lbl_val.setFont(font_w_val)
            lbl_val.setStyleSheet("color: #00e676; border: none;")
            self.weather_calib_labels[name] = lbl_val
            
            cell_layout.addWidget(lbl_title)
            cell_layout.addWidget(lbl_val)
            grid_layout.addWidget(cell_widget, row, col)
            
        left_layout.addWidget(grid_widget)
        
        # 底部分隔线与时间戳
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border: 1px solid #2d324f;")
        left_layout.addSpacing(15)
        left_layout.addWidget(line)
        left_layout.addSpacing(15)
        
        self.lbl_calib_timestamp = QLabel("气象设备时间戳: 等待同步...")
        self.lbl_calib_timestamp.setFont(QFont("Consolas", 14, QFont.Bold))
        self.lbl_calib_timestamp.setStyleSheet("color: #f39c12; border: none;")
        self.lbl_calib_timestamp.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_calib_timestamp)
        
        layout.addWidget(left_panel, 6)
        
        # === 右侧：控制按钮 ===
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignVCenter)
        right_layout.setSpacing(30)
        
        btn_style = """
            QPushButton { background-color: #2962ff; color: white; border: none; border-radius: 8px; padding: 20px; }
            QPushButton:hover { background-color: #0039cb; }
            QPushButton:pressed { background-color: #00227b; }
        """
        btn_font = QFont("Arial", 16, QFont.Bold)
        
        self.btn_sync_clock = QPushButton("时钟同步校准")
        self.btn_sync_clock.setFont(btn_font)
        self.btn_sync_clock.setStyleSheet(btn_style)
        
        self.btn_zero_wind = QPushButton("风速调零校准")
        self.btn_zero_wind.setFont(btn_font)
        self.btn_zero_wind.setStyleSheet(btn_style)
        
        right_layout.addWidget(self.btn_sync_clock)
        right_layout.addWidget(self.btn_zero_wind)
        
        layout.addWidget(right_panel, 4)
        self.stack.addWidget(page)

    def init_page_5_db_browser(self):
        """历史数据浏览面板 (Index 5)"""
        page = QWidget()
        page.setStyleSheet("background-color: #0f111a;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)

        # 顶部控制栏
        top_layout = QHBoxLayout()
        title = QLabel("车辆宏观监测数据浏览")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffffff; border: none;")

        # 采集任务列表的下拉菜单
        self.session_combo = QComboBox()
        self.session_combo.setFont(QFont("Arial", 12))
        self.session_combo.setMinimumWidth(280)
        self.session_combo.setStyleSheet("""
            QComboBox { background-color: #1a1d2d; color: white; border: 1px solid #2d324f; border-radius: 5px; padding: 5px; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #1a1d2d; color: white; selection-background-color: #2962ff; }
        """)

        self.btn_refresh_db = QPushButton(" 刷新数据 ")
        self.btn_refresh_db.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_refresh_db.setStyleSheet("background-color: #f39c12; color: white; border-radius: 5px; padding: 10px;")

        top_layout.addWidget(title)
        top_layout.addSpacing(20)
        top_layout.addWidget(self.session_combo)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_refresh_db)
        layout.addLayout(top_layout)
        layout.addSpacing(15)

        # 数据库表格控件
        self.db_table = QTableWidget()
        self.db_table.setColumnCount(8)
        # 设置表头字段定义
        self.db_table.setHorizontalHeaderLabels([
            "目标 ID", "车型", "能源类型", "入场时间", "离场时间", "均速(m/s)", "主导工况", "结算状态"
        ])

        # 表格样式调整，适应深色主题
        self.db_table.setStyleSheet("""
            QTableWidget { background-color: #1a1d2d; color: #ffffff; gridline-color: #2d324f; border: 1px solid #2d324f; }
            QHeaderView::section { background-color: #2962ff; color: white; font-weight: bold; padding: 5px; border: 1px solid #2d324f; }
        """)
        self.db_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.db_table.setEditTriggers(QTableWidget.NoEditTriggers) # 禁止双击编辑
        self.db_table.setSelectionBehavior(QTableWidget.SelectRows) # 整行选中

        layout.addWidget(self.db_table)
        self.stack.addWidget(page)

    def closeEvent(self, event):
        """窗口关闭时，转交 Controller 处理清理工作"""
        if hasattr(self, 'close_callback') and self.close_callback:
            self.close_callback()
        event.accept()
    