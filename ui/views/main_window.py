from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QStackedWidget, QTabWidget, 
                             QGridLayout, QFrame, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QComboBox, QSlider, QRadioButton,
                             QButtonGroup)
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
        # === 全局纯黑底色与白字基调 ===
        self.setStyleSheet("QMainWindow { background-color: #000000; } QWidget { color: #ffffff; }")

        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: #000000;") # 确保中间层也是黑的
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # === 统一定义镂空（Hollow）样式的 CSS ===
        # 常规白色镂空
        self.style_hollow_white = """
            QPushButton { background-color: transparent; border: 2px solid #ffffff; color: #ffffff; border-radius: 6px; padding: 10px; }
            QPushButton:hover { background-color: rgba(255, 255, 255, 0.1); }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 0.2); }
        """
        # 绿色运行/确认镂空
        self.style_hollow_green = """
            QPushButton { background-color: transparent; border: 2px solid #00e676; color: #00e676; border-radius: 6px; padding: 10px; }
            QPushButton:hover { background-color: rgba(0, 230, 118, 0.1); }
            QPushButton:pressed { background-color: rgba(0, 230, 118, 0.2); }
        """
        # 红色警告/停止镂空
        self.style_hollow_red = """
            QPushButton { background-color: transparent; border: 2px solid #ff4d4f; color: #ff4d4f; border-radius: 6px; padding: 10px; }
            QPushButton:hover { background-color: rgba(255, 77, 79, 0.1); }
            QPushButton:pressed { background-color: rgba(255, 77, 79, 0.2); }
        """

        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)

        self.nav_widget = QWidget()
        self.nav_layout = QHBoxLayout(self.nav_widget)
        self.nav_layout.setContentsMargins(20, 10, 20, 10)
        
        font = QFont("Arial", 16, QFont.Bold)

        # 应用镂空样式到底部导航栏
        self.btn_home = QPushButton("返回主页")
        self.btn_home.setFont(font); self.btn_home.setMinimumHeight(50)
        self.btn_home.setStyleSheet(self.style_hollow_white) 

        self.btn_stop = QPushButton("结束采集")
        self.btn_stop.setFont(font); self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet(self.style_hollow_red) 

        self.btn_delete_db = QPushButton("批量删除")
        self.btn_delete_db.setFont(font); self.btn_delete_db.setMinimumHeight(50)
        self.btn_delete_db.setStyleSheet(self.style_hollow_red)
        self.btn_delete_db.setVisible(False)

        self.btn_prev = QPushButton("◀ 上一步")
        self.btn_prev.setFont(font); self.btn_prev.setMinimumHeight(50)
        self.btn_prev.setStyleSheet(self.style_hollow_white)
        
        self.btn_next = QPushButton("下一步 ▶")
        self.btn_next.setFont(font); self.btn_next.setMinimumHeight(50)
        self.btn_next.setStyleSheet(self.style_hollow_white)

        self.nav_layout.addWidget(self.btn_home)
        self.nav_layout.addWidget(self.btn_stop)
        self.nav_layout.addWidget(self.btn_delete_db)
        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.btn_next)
        self.main_layout.addWidget(self.nav_widget)

        # 初始化页面：按顺序压入栈中
        self.init_page_main_menu()     # 主界面
        self.init_page_calibration()   # 透视标定步骤
        self.init_page_size_settings() # 尺度设置步骤
        self.init_page_pos_settings()  # 方位设置步骤
        self.init_page_monitor()       # 运行面板
        self.init_page_weather_calib() # 气象站校准页面
        self.init_page_db_browser()    # 数据库浏览页面
        self.init_page_settings()      # 总设置界面

    def init_page_main_menu(self):
        """主调度界面"""
        self.page_main_menu = QWidget()
        self.page_main_menu.setStyleSheet("background-color: #0f111a;") # 深邃的边缘计算科技蓝/黑底色
        layout = QHBoxLayout(self.page_main_menu)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(30)

        # --- 左侧：硬件与状态信息看板 (BIOS 仪表盘风格) ---
        left_panel = QFrame()
        # 替换为灰黑底色和暗色边框
        left_panel.setStyleSheet("background-color: #181818; border: 1px solid #444; border-radius: 12px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(25, 25, 25, 25)

        title = QLabel("系统状态")
        title.setFont(QFont("Consolas", 18, QFont.Bold))
        title.setStyleSheet("color: #ffffff; border: none;") # 标题用纯白
        left_layout.addWidget(title)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border: 1px solid #333;") # 暗色分割线
        left_layout.addWidget(line)
        left_layout.addSpacing(15)

        self.status_labels = {}
        status_keys = ["系统时间", "存储空间", "网络连接", "气象网关", "CPU 温度", "NPU 温度"]

        font_label = QFont("Consolas", 14)
        font_value = QFont("Consolas", 14, QFont.Bold)
        
        # 循环创建 UI 组件，并将值 Label 存入字典
        for key in status_keys:
            row = QHBoxLayout()
            
            lbl_k = QLabel(key)
            lbl_k.setStyleSheet("color: #aaaaaa; border: none;") # 键名：暗灰，降低视觉干扰
            lbl_k.setFont(font_label)
            
            lbl_v = QLabel("--")  
            lbl_v.setStyleSheet("color: #ffffff; border: none;") # 键值：明亮白，突出数据本身
            lbl_v.setFont(font_value)
            lbl_v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
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

        self.btn_app1 = QPushButton("多源数据采集")
        self.btn_app1.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_app1.setStyleSheet(self.style_hollow_white)

        self.btn_app2 = QPushButton("气象设备校准")
        self.btn_app2.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_app2.setStyleSheet(self.style_hollow_white)

        self.btn_app3 = QPushButton("浏览历史数据")
        self.btn_app3.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_app3.setStyleSheet(self.style_hollow_white)

        self.btn_exit = QPushButton("退出程序")
        self.btn_exit.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_exit.setStyleSheet(self.style_hollow_red)

        right_layout.addWidget(self.btn_app1)
        right_layout.addSpacing(15)
        right_layout.addWidget(self.btn_app2)
        right_layout.addSpacing(15)
        right_layout.addWidget(self.btn_app3)
        right_layout.addStretch()
        
        # 创建底部水平布局
        bottom_btn_layout = QHBoxLayout()
        bottom_btn_layout.setContentsMargins(0, 0, 0, 0)
        
        # 新增“设置”按钮
        self.btn_settings = QPushButton("设置")
        self.btn_settings.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_settings.setStyleSheet(self.style_hollow_white) # 使用统一的白色镂空样式
        
        self.btn_exit = QPushButton("退出")
        self.btn_exit.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_exit.setStyleSheet(self.style_hollow_red)

        # 两个按钮等宽并排，中间加一点间距
        bottom_btn_layout.addWidget(self.btn_settings)
        bottom_btn_layout.addSpacing(15)
        bottom_btn_layout.addWidget(self.btn_exit)

        # 将水平布局加入右侧主布局
        right_layout.addLayout(bottom_btn_layout)

        layout.addWidget(right_panel, 4) # 右侧占比 4

        self.stack.addWidget(self.page_main_menu)

    def init_page_settings(self):
        """系统设置面板"""
        self.page_settings = QWidget()
        self.page_settings.setStyleSheet("background-color: #0f111a;")
        layout = QVBoxLayout(self.page_settings)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("系统设置与参数配置")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #ffffff; border: none;")
        layout.addWidget(title)
        layout.addSpacing(15)

        # 创建多页签容器
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setFont(QFont("Arial", 14))
        # 适配深色科技风的 Tab 样式
        self.settings_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; background: #181818; border-radius: 5px; }
            QTabBar::tab { background: #222; color: #aaaaaa; padding: 10px 25px; border-top-left-radius: 5px; border-top-right-radius: 5px; margin-right: 2px; }
            QTabBar::tab:selected { background: #181818; color: #ffffff; font-weight: bold; border: 1px solid #444; border-bottom: none; }
        """)
        layout.addWidget(self.settings_tabs)

        # --- 页签 1: 视频源设置 ---
        tab_video_source = QWidget()
        vs_layout = QVBoxLayout(tab_video_source)
        vs_layout.setContentsMargins(40, 40, 40, 40)

        lbl_desc = QLabel("选择输入视频流：")
        lbl_desc.setFont(QFont("Arial", 14))
        lbl_desc.setStyleSheet("color: #cccccc; border: none;")
        vs_layout.addWidget(lbl_desc)
        vs_layout.addSpacing(30)

        # 创建互斥按钮组
        self.source_btn_group = QButtonGroup(self.page_settings)

        # --- 选项 1: 本地视频 ---
        local_layout = QHBoxLayout()
        
        self.radio_source_local = QRadioButton("本地视频文件")
        self.radio_source_local.setFont(QFont("Arial", 16))
        self.radio_source_local.setStyleSheet("color: white;")
        self.radio_source_local.setChecked(True) # 默认选中本地

        # 浏览按钮
        self.btn_browse_local = QPushButton(" 浏览文件 ")
        self.btn_browse_local.setFont(QFont("Arial", 14))
        self.btn_browse_local.setStyleSheet(self.style_hollow_white)
        self.btn_browse_local.setFixedHeight(40) # 控制一下高度不要太大

        # 路径显示标签 (初始加载 loader 中当前的配置路径)
        self.lbl_local_path = QLabel(cfg.LOCAL_VIDEO_PATH) 
        self.lbl_local_path.setFont(QFont("Arial", 12))
        self.lbl_local_path.setStyleSheet("color: #aaaaaa; border: none;")

        # 将组件拼装进水平布局
        local_layout.addWidget(self.radio_source_local)
        local_layout.addSpacing(15)
        local_layout.addWidget(self.btn_browse_local)
        local_layout.addSpacing(15)
        local_layout.addWidget(self.lbl_local_path)
        local_layout.addStretch() # 靠左对齐，剩余空间留白

        vs_layout.addLayout(local_layout) # 加入主布局

        vs_layout.addSpacing(20)

        # --- 选项 2: 接入的摄像头 ---
        camera_layout = QHBoxLayout()
        
        self.radio_source_camera = QRadioButton("接入的摄像头")
        self.radio_source_camera.setFont(QFont("Arial", 16))
        self.radio_source_camera.setStyleSheet("color: white;")
        # 根据配置文件初始化选中状态
        if cfg.USE_CAMERA: self.radio_source_camera.setChecked(True)

        # 自动检测按钮
        self.btn_detect_camera = QPushButton(" 自动检测 ")
        self.btn_detect_camera.setFont(QFont("Arial", 14))
        self.btn_detect_camera.setStyleSheet(self.style_hollow_white)
        self.btn_detect_camera.setFixedHeight(40)

        # 设备信息显示
        self.lbl_camera_info = QLabel("未检测到设备")
        if cfg.USE_CAMERA: self.lbl_camera_info.setText("已接入默认摄像头")
        self.lbl_camera_info.setFont(QFont("Arial", 12))
        self.lbl_camera_info.setStyleSheet("color: #aaaaaa; border: none;")

        camera_layout.addWidget(self.radio_source_camera)
        camera_layout.addSpacing(15)
        camera_layout.addWidget(self.btn_detect_camera)
        camera_layout.addSpacing(15)
        camera_layout.addWidget(self.lbl_camera_info)
        camera_layout.addStretch()

        vs_layout.addLayout(camera_layout)
        vs_layout.addStretch()

        # 将两个单选框加入按钮组，强制互斥
        self.source_btn_group.addButton(self.radio_source_local)
        self.source_btn_group.addButton(self.radio_source_camera)

        # 统一在这里根据配置进行且仅进行一次选中判定
        if cfg.USE_CAMERA:
            self.radio_source_camera.setChecked(True)
        else:
            self.radio_source_local.setChecked(True)

        self.settings_tabs.addTab(tab_video_source, "视频源配置")

        self.stack.addWidget(self.page_settings)

    def init_page_calibration(self):
        self.page_calibration = QWidget()
        layout = QVBoxLayout(self.page_calibration)
        
        self.lbl_calib_title = QLabel("步骤 1/3: 拖拽 4 个角点进行标定")
        self.lbl_calib_title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(self.lbl_calib_title)
        
        self.canvas = CalibrationCanvas()
        self.canvas.load_frame(cfg.VIDEO_PATH) # 加载第一帧
        layout.addWidget(self.canvas)
        
        self.stack.addWidget(self.page_calibration)
    
    def init_page_size_settings(self):
        self.page_size_settings = QWidget()
        layout = QVBoxLayout(self.page_size_settings)
        layout.setAlignment(Qt.AlignCenter)
        
        self.lbl_settings_title = QLabel("步骤 2/3: 设置真实物理尺寸")
        self.lbl_settings_title.setFont(QFont("Arial", 18, QFont.Bold))
        self.lbl_settings_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_settings_title)

        # 粗微调控制器
        def create_adjuster(label_text, init_value, setter_func):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFont(QFont("Arial", 16))
            lbl.setMinimumWidth(220)
            
            btn_minus_coarse = QPushButton("- 1.0")
            btn_minus_fine = QPushButton("- 0.1")
            btn_plus_fine = QPushButton("+ 0.1")
            btn_plus_coarse = QPushButton("+ 1.0")
            
            for btn in [btn_minus_coarse, btn_minus_fine, btn_plus_fine, btn_plus_coarse]:
                btn.setFixedSize(65, 50)
                btn.setStyleSheet(self.style_hollow_white)
            
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

        self.stack.addWidget(self.page_size_settings)

    def init_page_pos_settings(self):
        """物理先验参数与气象站位置设置"""
        self.page_pos_settings = QWidget()
        layout = QVBoxLayout(self.page_pos_settings)
        layout.setContentsMargins(30, 20, 30, 20)
        
        title = QLabel("步骤 3/3: 物理与环境先验参数")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; padding: 5px;")
        layout.addWidget(title)
        layout.addSpacing(15)

        # 改为水平布局：左侧气象站，右侧方位角旋钮
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # ==========================================
        # 1. (左侧) 气象站空间位置设置
        # ==========================================
        wx_frame = QFrame()
        wx_frame.setMinimumHeight(320)
        wx_frame.setStyleSheet("background-color: #181818; border: 1px solid #444; border-radius: 10px; padding: 15px 15px 25px 15px;")
        wx_layout = QVBoxLayout(wx_frame)
        wx_layout.setSpacing(15)
        
        lbl_wx_title = QLabel("气象站部署位置 (相较于道路):")
        lbl_wx_title.setFont(QFont("Arial", 14, QFont.Bold))
        lbl_wx_title.setStyleSheet("color: #00e676; border: none; padding: 2px;")
        wx_layout.addWidget(lbl_wx_title)

        lbl_side = QLabel("所处方位:")
        lbl_side.setFont(QFont("Arial", 14))
        lbl_side.setStyleSheet("color: white; border: none; padding: 2px;")
        wx_layout.addWidget(lbl_side)
        
        self.combo_wx_side = QComboBox()
        self.combo_wx_side.addItems(["道路左侧 (x 坐标 ≤ 0)", "道路右侧 (x 坐标 ≥ 车道总宽)"])
        self.combo_wx_side.setFont(QFont("Arial", 13))
        # 增加 min-height 和 padding 解决下拉框文字边缘被裁切的问题
        self.combo_wx_side.setStyleSheet("background-color: #000; color: white; border: 1px solid #555; padding: 8px; min-height: 25px;")
        wx_layout.addWidget(self.combo_wx_side)
        
        wx_layout.addSpacing(10)

        lbl_dist = QLabel("距路缘距离 (m):")
        lbl_dist.setFont(QFont("Arial", 14))
        lbl_dist.setStyleSheet("color: white; border: none; padding: 2px;")
        wx_layout.addWidget(lbl_dist)

        self.wx_dist_to_edge = 0.0
        
        dist_layout = QHBoxLayout()
        btn_minus = QPushButton("- 0.1")
        btn_plus = QPushButton("+ 0.1")
        for btn in [btn_minus, btn_plus]:
            # 增加高度到 45，防止大字体按钮文字上下被一刀切
            btn.setFixedSize(80, 45)
            btn.setStyleSheet(self.style_hollow_white)
        
        val_lbl = QLabel(f"{self.wx_dist_to_edge:.1f} m")
        val_lbl.setFont(QFont("Arial", 20, QFont.Bold))
        val_lbl.setStyleSheet("color: white; border: none; padding: 2px;")
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setMinimumWidth(80)
        
        state = {'val': self.wx_dist_to_edge}
        def make_callback(delta):
            def callback():
                # 必须用 round 包裹，修复 Python 浮点数相加引发的无限长小数尾巴乱码
                state['val'] = round(max(0.0, state['val'] + delta), 1)
                val_lbl.setText(f"{state['val']:.1f} m")
                self.wx_dist_to_edge = state['val']
            return callback

        btn_minus.clicked.connect(make_callback(-0.1))
        btn_plus.clicked.connect(make_callback(0.1))
        
        dist_layout.addStretch()
        dist_layout.addWidget(btn_minus)
        dist_layout.addSpacing(10)
        dist_layout.addWidget(val_lbl)
        dist_layout.addSpacing(10)
        dist_layout.addWidget(btn_plus)
        dist_layout.addStretch()
        
        wx_layout.addLayout(dist_layout)
        wx_layout.addStretch() # 把组件向上顶，让排版更紧凑
        content_layout.addWidget(wx_frame)

        # ==========================================
        # 2. (右侧) 道路方向角设置 (圆盘旋钮)
        # ==========================================
        road_frame = QFrame()
        road_frame.setMinimumHeight(320)
        road_frame.setStyleSheet("background-color: #181818; border: 1px solid #444; border-radius: 10px; padding: 15px 15px 25px 15px;")
        road_layout = QVBoxLayout(road_frame)
        road_layout.setSpacing(10)
        
        lbl_road_title = QLabel("道路走向方位角 (正北为0°):")
        lbl_road_title.setFont(QFont("Arial", 14, QFont.Bold))
        lbl_road_title.setStyleSheet("color: #00e676; border: none; padding: 2px;")
        road_layout.addWidget(lbl_road_title)

        lbl_info = QLabel(
            "顺着设定的道路矢量方向看，\n"
            "须确保【气象站位于道路右侧】。\n"
            "示例：设备在路东，监测南北向道路，\n"
            "则矢量应定为由南向北（0°）。"
        )
        lbl_info.setFont(QFont("Arial", 11))
        lbl_info.setWordWrap(True) # 允许长句子自动换行
        lbl_info.setStyleSheet("color: #ff9800; border: none; padding: 12px; background-color: rgba(255,152,0, 0.1); border-radius: 5px;")
        road_layout.addWidget(lbl_info)
        road_layout.addSpacing(10)

        dial_layout = QHBoxLayout()
        
        # 引入 PyQt5 原生的 QDial 圆盘组件替代滑动条
        from PyQt5.QtWidgets import QDial
        self.slider_road_angle = QDial() # 仍叫 slider_road_angle，免去修改 Controller 的麻烦
        self.slider_road_angle.setRange(0, 359)
        self.slider_road_angle.setSingleStep(5)
        self.slider_road_angle.setPageStep(15)
        self.slider_road_angle.setWrapping(True) # 开启循环（355 往上滑直接变 0）
        self.slider_road_angle.setNotchesVisible(True) # 显示刻度线
        self.slider_road_angle.setFixedSize(130, 130)
        
        # 赋予 QDial 赛博朋克的深黑底色
        self.slider_road_angle.setStyleSheet("""
            QDial { background-color: #111; }
        """)
        
        self.lbl_angle_val = QLabel("0°")
        self.lbl_angle_val.setFont(QFont("Arial", 26, QFont.Bold))
        self.lbl_angle_val.setStyleSheet("color: white; border: none; padding: 5px;")
        self.lbl_angle_val.setAlignment(Qt.AlignCenter)
        self.lbl_angle_val.setMinimumWidth(90)
        
        # === 拦截信号实现 5 度强制吸附 (Snapping) ===
        def snap_to_5_degrees(v):
            snapped_val = int(round(v / 5.0) * 5) % 360
            if self.slider_road_angle.value() != snapped_val:
                self.slider_road_angle.blockSignals(True)
                self.slider_road_angle.setValue(snapped_val)
                self.slider_road_angle.blockSignals(False)
            self.lbl_angle_val.setText(f"{snapped_val}°")

        self.slider_road_angle.valueChanged.connect(snap_to_5_degrees)
        snap_to_5_degrees(0) # 初始化显示
        
        dial_layout.addStretch()
        dial_layout.addWidget(self.slider_road_angle)
        dial_layout.addSpacing(25)
        dial_layout.addWidget(self.lbl_angle_val)
        dial_layout.addStretch()
        
        road_layout.addLayout(dial_layout)
        road_layout.addSpacing(20)
        road_layout.addStretch()
        content_layout.addWidget(road_frame)

        layout.addLayout(content_layout)
        self.stack.addWidget(self.page_pos_settings)

    def init_page_monitor(self):
        self.page_monitor = QWidget()
        layout = QVBoxLayout(self.page_monitor)
        layout.setContentsMargins(0,0,0,0)
        
        # 创建底部 Tab 栏
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.South) # 页签放在底部
        self.tabs.setFont(QFont("Arial", 14, QFont.Bold))
        
        # 应用自适应底部 Tab 的深色科技风样式
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #444; 
                background: #181818; 
                border-radius: 5px; 
            }
            QTabBar::tab { 
                background: #222; 
                color: #aaaaaa; 
                padding: 10px 25px; 
                border-bottom-left-radius: 5px; 
                border-bottom-right-radius: 5px; 
                margin-right: 2px; 
            }
            QTabBar::tab:selected { 
                background: #181818; 
                color: #ffffff; 
                font-weight: bold; 
                border: 1px solid #444; 
                border-top: none; 
            }
        """)

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
        weather_left_panel.setStyleSheet("background-color: #181818; border: 1px solid #444; border-radius: 12px;")
        w_grid_layout = QGridLayout(weather_left_panel)
        w_grid_layout.setContentsMargins(25, 25, 25, 25)
        w_grid_layout.setSpacing(20)
        
        self.weather_monitor_labels = {}
        weather_items = [("温度", "°C"), ("湿度", "%"), ("风速", "m/s"), ("风向", "°"), ("PM2.5", "μg/m³"), ("PM10", "μg/m³")]
        
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
            lbl_title.setStyleSheet("color: #aaaaaa; border: none;") # 键名去色暗化
            
            lbl_val = QLabel(f"-- {unit}")
            lbl_val.setFont(font_w_val)
            lbl_val.setStyleSheet("color: #ffffff; border: none;") # 键值提亮
            lbl_val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            self.weather_monitor_labels[name] = lbl_val 
            
            cell_layout.addWidget(lbl_title)
            cell_layout.addWidget(lbl_val)
            cell_layout.addStretch()
            w_grid_layout.addWidget(cell_widget, row, col)
            
        w_layout.addWidget(weather_left_panel, 7)
        
        # === 右侧：气象站插画展示区 ===
        weather_right_panel = QFrame()
        weather_right_panel.setStyleSheet("background-color: #181818; border: 1px solid #444; border-radius: 12px;")
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
        
        self.stack.addWidget(self.page_monitor)
    
    def init_page_weather_calib(self):
        """气象站独立校准面板"""
        self.page_weather_calib = QWidget()
        self.page_weather_calib.setStyleSheet("background-color: #0f111a;")
        layout = QHBoxLayout(self.page_weather_calib)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(30)
        
        # === 左侧：气象数据看板 + 时间戳 ===
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #181818; border: 1px solid #444; border-radius: 12px;")
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
            lbl_title.setStyleSheet("color: #aaaaaa; border: none;")
            lbl_val = QLabel(f"-- {unit}")
            lbl_val.setFont(font_w_val)
            lbl_val.setStyleSheet("color: #ffffff; border: none;")
            self.weather_calib_labels[name] = lbl_val
            
            cell_layout.addWidget(lbl_title)
            cell_layout.addWidget(lbl_val)
            grid_layout.addWidget(cell_widget, row, col)
            
        left_layout.addWidget(grid_widget)
        
        # 底部分隔线与时间戳
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("border: 1px solid #333;")
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
        
        btn_font = QFont("Arial", 16, QFont.Bold)
        
        self.btn_sync_clock = QPushButton("时钟同步校准")
        self.btn_sync_clock.setFont(btn_font)
        self.btn_sync_clock.setStyleSheet(self.style_hollow_white)
        
        self.btn_zero_wind = QPushButton("风速调零校准")
        self.btn_zero_wind.setFont(btn_font)
        self.btn_zero_wind.setStyleSheet(self.style_hollow_white)
        
        right_layout.addWidget(self.btn_sync_clock)
        right_layout.addWidget(self.btn_zero_wind)
        
        layout.addWidget(right_panel, 4)
        self.stack.addWidget(self.page_weather_calib)

    def init_page_db_browser(self):
        """历史数据浏览面板"""
        self.page_db_browser = QWidget()
        # 统一为纯黑背景
        self.page_db_browser.setStyleSheet("background-color: #000000;")
        layout = QVBoxLayout(self.page_db_browser)
        layout.setContentsMargins(30, 30, 30, 30)

        # 顶部控制栏
        top_layout = QHBoxLayout()
        title = QLabel("车辆宏观监测数据浏览")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffffff; border: none;")

        # 采集任务列表的下拉菜单 (去色，改为灰黑风格)
        self.session_combo = QComboBox()
        self.session_combo.setFont(QFont("Arial", 12))
        self.session_combo.setMinimumWidth(280)
        self.session_combo.setStyleSheet("""
            QComboBox { background-color: #111111; color: white; border: 1px solid #444; border-radius: 5px; padding: 5px; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #111111; color: white; selection-background-color: #333333; }
        """)

        # 应用绿色镂空样式
        self.btn_refresh_db = QPushButton(" 刷新数据 ")
        self.btn_refresh_db.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_refresh_db.setStyleSheet(self.style_hollow_green)

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

        # 表格样式调整：彻底去色，表头变暗，选中行变灰
        self.db_table.setStyleSheet("""
            QTableWidget { 
                background-color: #111111; 
                color: #ffffff; 
                gridline-color: #333333; 
                border: 1px solid #444444; 
            }
            QTableWidget::item:selected {
                background-color: #333333; 
                color: #ffffff;
            }
            QHeaderView::section { 
                background-color: #181818; 
                color: #aaaaaa; 
                font-weight: bold; 
                padding: 5px; 
                border: 1px solid #333333; 
            }
            QTableCornerButton::section {
                background-color: #181818;
                border: 1px solid #333333;
            }
        """)
        self.db_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.db_table.setEditTriggers(QTableWidget.NoEditTriggers) # 禁止双击编辑
        self.db_table.setSelectionBehavior(QTableWidget.SelectRows) # 整行选中

        layout.addWidget(self.db_table)
        self.stack.addWidget(self.page_db_browser)

    def closeEvent(self, event):
        """窗口关闭时，转交 Controller 处理清理工作"""
        if hasattr(self, 'close_callback') and self.close_callback:
            self.close_callback()
        event.accept()
