import os
import json
import cv2
import numpy as np
import infra.config.loader as cfg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QTableWidgetItem, QVBoxLayout, 
                             QHBoxLayout, QLabel, 
                             QPushButton, QRadioButton,
                             QFileDialog)
from PyQt5.QtGui import QImage, QPixmap, QFont
from datetime import datetime
from ui.components.edge_dialog import EdgeMessageBox, EdgeAnimatedDialog, EdgeExportDialog
from infra.sys.sys_monitor import SysMonitor
from infra.store.sqlite_manager import DatabaseManager
from infra.store.storage_manager import StorageManager
from perception.gst_pipeline import GstPipelineManager
import perception.gst_pipeline as gst

class MainController:
    """!
    @brief 主控制器类，遵循标准的 MVC 架构模式。
    @details 负责统筹前端视图与后端视觉/物理计算引擎的交互，管理系统状态机并处理组件间的信号分发。
    控制器不负责具体的物理计算、数据库连接或硬件控制，所有底层设施均通过依赖注入(DI)方式接入。
    """
    def __init__(self, view, components: dict):
        """!
        @brief 初始化控制器实例并装配系统。
        @param view 绑定的主窗口视图实例 (MainWindow)。
        @param components 由 Bootstrap 工厂组装的系统组件与服务字典。
        """
        self.view = view
        
        # ==========================================
        # 1. 依赖解包 (Dependency Injection Unpacking)
        # ==========================================
        # 将传入的黑盒字典拆解为控制器直接需要的具体组件
        self._unpack_components(components)

        # ==========================================
        # 2. 状态机管理 (State Management)
        # ==========================================
        # 记录当前业务的运行状态
        self.is_collecting = False
        self.sampled_tid = None
        
        # 多线程/跨进程控制句柄
        self.worker = None
        self.global_camera = None

        # ==========================================
        # 3. UI 路由与信号绑定 (View Routing & Binding)
        # ==========================================
        # 定义显式的页面向导流，严格控制步骤流转
        self.wizard_flow = [
            self.view.page_calibration,       # 步骤 1: 视觉标定
            self.view.page_size_settings,     # 步骤 2: 道路 ROI 尺寸设置
            self.view.page_pos_settings,      # 步骤 3: 仪器与道路方位设置
            self.view.page_monitor            # 步骤 4: 实时监控屏
        ]
        self.bind_signals()

        # ==========================================
        # 4. 后台守护任务 (Background Timers)
        # ==========================================
        # 统一管理需要定时执行的轻量级 UI 刷新任务
        self._init_timers()

        # 在刷新状态前，显式将页面设置为主菜单
        self.view.stack.setCurrentWidget(self.view.page_main_menu)
        self.update_nav_buttons()

    def _unpack_components(self, components: dict):
        """!
        @brief 解析并挂载底层组件，包含防御性校验。
        """
        # 保存原始字典以备不时之需
        self.components = components
        
        # 提取业务领域模型与数据库
        self.cfg = components.get('config')
        self.db = components.get('db')
        self.registry = components.get('registry')
        self.visualizer = components.get('visualizer')
        self.weather_gw = components.get('weather_station')
        
        # 提取跨进程通信的核心基础设施
        self.sync_queue = components.get('sync_queue')
        self.stop_event = components.get('stop_event')

        # 防御性断言：确保关键 IPC 组件不可或缺
        if not self.sync_queue or not self.stop_event:
            raise ValueError("[Controller] 致命错误：缺失必要的跨进程通信管道 (Queue) 或控制信号 (Event)。")

    def start_monitoring(self):
        """!
        @brief 启动监控流程（已取代旧版 start_engine）：装配物理参数 -> 释放硬件 -> 启动引擎。
        """
        if self.is_collecting: return

        # 任务会话 (Session) 创建
        import time
        from datetime import datetime
        
        # 生成基于时间戳的唯一 Session ID
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 将 Session ID 同步给全局配置，底层引擎将依赖它进行落盘
        self.cfg.CURRENT_SESSION_ID = self.current_session_id

        # 在数据库中注册这条主任务记录
        if hasattr(self, 'db') and self.db:
            location_desc = getattr(self.cfg, 'LOCATION_DESC', "默认监控点")
            self.db.create_session(self.current_session_id, time.time(), location_desc)

        # --- 1. 物理先验参数同步 ---
        # 从 UI 层获取透视标定点和物理尺寸，同步到全局配置中，供后端的对齐引擎(AlignmentEngine)使用
        self.cfg.SOURCE_POINTS = self.view.canvas.get_real_points().tolist()
        self.cfg.PHYS_WIDTH = self.view.phys_w
        self.cfg.PHYS_HEIGHT = self.view.phys_h

        # 同步更新字典里的 numpy 数组，供底层引擎使用
        import numpy as np
        self.components['norm_source_points'] = np.array(self.cfg.SOURCE_POINTS, dtype=np.float32)

        # 计算真实的物理目标点矩阵
        self.components['target_points'] = np.array([
            [0, self.cfg.PHYS_HEIGHT],
            [self.cfg.PHYS_WIDTH, self.cfg.PHYS_HEIGHT],
            [self.cfg.PHYS_WIDTH, 0],
            [0, 0]
        ], dtype=np.float32)

        # --- 2. 硬件控制权移交 ---
        # 释放主界面标定画布占用的摄像头流，为感知子进程或后台推理让路
        if getattr(self, 'global_camera', None):
            print("[Controller] 正在释放主进程摄像头管线，为后端推理引擎让路...")
            self.global_camera.stop()
            self.global_camera = None

        from app.monitor_engine import TrafficMonitorEngine
        from ui.workers.engine_worker import EngineWorker

        # --- 3. 实例化 Worker 并绑定 UI 刷新回调 ---
        self.worker = EngineWorker()
        self.worker.frame_ready.connect(self.on_frame_received)

        # --- 4. 实例化后端核心引擎 ---
        # 使用依赖注入(DI)的方式，将所需的所有资源一次性传给引擎
        self.engine = TrafficMonitorEngine(
            config=self.cfg,
            components=self.components,
            sync_queue=self.sync_queue,
            frame_callback=self.worker.emit_frame 
        )

        # --- 5. 启动后台守护线程 ---
        self.worker.set_engine(self.engine)
        self.worker.start()
        
        self.is_collecting = True
        
        # --- 6. 刷新界面状态 ---
        self.update_nav_buttons()
        self.update_main_menu_btn_style()
        print(">>> [Controller] 监控引擎已全面启动。")

    def on_frame_received(self, rgb_frame):
        """!
        @brief UI 线程回调：渲染图像、刷新数据面板并释放 Worker 锁。
        """
        try:
            # 1. 无头模式拦截：如果开启了隐藏画面，则跳过 UI 图像的转换与渲染，节省 CPU
            if not getattr(self, 'is_headless', False):
                if hasattr(self.view, 'video_canvas'):
                    self.view.video_canvas.update_image(rgb_frame)
            
        except Exception as e:
            print(f"[UI异常] 视频帧渲染失败: {e}")
            
        finally:
            # 3. 核心安全机制：无论是否无头模式，无论渲染是否报错，
            # 必须调用新版 Worker 的 unlock_frame 释放锁，否则后台引擎会被永久挂起
            if self.worker:
                self.worker.unlock_frame()

    def on_engine_finished(self):
        """!
        @brief 引擎结束的收尾工作。
        @details 负责彻底清空内存对象引用、重置控制器状态机，并恢复 UI 界面。
        """
        # 1. 彻底断开与销毁底层对象引用
        self.worker = None
        self.engine = None
        self.global_camera = None

        # 2. 重置控制器的核心状态机
        self.is_collecting = False
        self.sampled_tid = None
        self.current_session_id = None

        # 3. 恢复 UI 组件的初始状态 (为下次采集做准备)
        self.view.btn_stop.setText("结束采集")
        self.view.btn_stop.setEnabled(True)
        self.view.btn_stop.setStyleSheet(self.view.style_hollow_red) # 恢复红色镂空
        
        # 4. 刷新主菜单样式并路由回主页
        self.update_main_menu_btn_style()
        self.return_to_home()
        
        print(">>> [Controller] 监控任务已完全终止并回收所有资源。")

    def _init_timers(self):
        """初始化 UI 层的定时刷新器与守护任务"""
        # 1. 恢复系统硬件看板监控定时器 (1Hz)
        self.sys_timer = QTimer(self.view)
        self.sys_timer.timeout.connect(self.update_sys_board) 
        self.sys_timer.start(1000) 

        # 2. 恢复低频磁盘空间清理守护定时器 (每分钟检查)
        self.disk_monitor_timer = QTimer(self.view)
        self.disk_monitor_timer.timeout.connect(self.check_and_cleanup_disk_space)
        self.disk_monitor_timer.start(60000)

        # 3. 恢复“录制界面”的初始化显示（从配置文件读取并反显到 UI）
        self.view.btn_record_switch.setChecked(self.cfg.ENABLE_RECORD)
        self.handle_record_switch_toggled(self.cfg.ENABLE_RECORD)
        
        segment_map = {5: 0, 10: 1, 15: 2, 30: 3}
        self.view.combo_segment_time.setCurrentIndex(segment_map.get(self.cfg.RECORD_SEGMENT_MIN, 1))

        # 每分钟检查 SSD 剩余空间并清理旧视频
        self.disk_monitor_timer = QTimer(self.view)
        self.disk_monitor_timer.timeout.connect(self.check_and_cleanup_disk_space)
        self.disk_monitor_timer.start(60000) # 每分钟检查一次

        # UI 刷新定时器，专门负责热成像预览与车辆数据看板
        self.dash_timer = QTimer(self.view)
        self.dash_timer.timeout.connect(self.update_timer_tasks)
        self.dash_timer.start(100) # 10 Hz 刷新率

    def bind_signals(self):
        """将视图组件的事件绑定到控制器的逻辑上"""
        self.view.btn_home.clicked.connect(self.return_to_home)
        self.view.btn_stop.clicked.connect(self.stop_collection_trigger)
        self.view.btn_prev.clicked.connect(self.prev_page)
        self.view.btn_next.clicked.connect(self.next_page)
        self.view.btn_app1.clicked.connect(self.route_app1_click)
        self.view.btn_app2.clicked.connect(self.route_app2_click)
        self.view.btn_app3.clicked.connect(self.route_app3_click)
        self.view.btn_headless.clicked.connect(self.toggle_headless)

        # 绑定气象站的校准按钮事件
        self.view.btn_sync_clock.clicked.connect(self.handle_sync_clock)
        self.view.btn_zero_wind.clicked.connect(self.handle_zero_wind)

        # 数据表单的按钮/列表绑定
        self.view.btn_refresh_db.clicked.connect(self.handle_db_refresh)
        self.view.btn_delete_db.clicked.connect(self.show_batch_delete_dialog)
        self.view.btn_export_db.clicked.connect(self.handle_export_db_data)
        self.view.session_combo.currentIndexChanged.connect(self.update_db_table)

        # 设置界面相关按钮的绑定
        self.view.btn_settings.clicked.connect(self.route_settings_click)
        self.view.btn_browse_local.clicked.connect(self.handle_browse_local_video)
        self.view.btn_detect_camera.clicked.connect(self.handle_detect_camera)
        self.view.radio_source_local.toggled.connect(self.handle_source_type_changed)
        self.view.radio_source_camera.toggled.connect(self.handle_source_type_changed)
        self.view.radio_mode_inference.toggled.connect(self.handle_run_mode_changed)
        self.view.radio_mode_collection.toggled.connect(self.handle_run_mode_changed)
        self.view.btn_record_switch.toggled.connect(self.handle_record_switch_toggled)
        self.view.btn_export_videos.clicked.connect(self.handle_export_videos)
        self.view.combo_segment_time.currentIndexChanged.connect(self.save_record_settings)

        # 退出程序按钮的绑定
        self.view.btn_exit.clicked.connect(self.handle_exit_request)
    
    def handle_sync_clock(self):
        if not self.weather_gw: return
        dialog = EdgeMessageBox(self.view, "时钟同步校准", "确认与远端系统同步时间吗？")
        if dialog.exec_() == EdgeMessageBox.Accepted: # 捕获自定义关闭信号
            self.weather_gw.sync_time()
            print("前端控制器：已确认下发时钟同步指令")
            
    def handle_zero_wind(self):
        if not self.weather_gw: return
        dialog = EdgeMessageBox(self.view, "风速调零校准", "确认执行风速调零吗？", "请确保传感器处于无风环境，等待10秒后完成调零。", is_warning=True)
        if dialog.exec_() == EdgeMessageBox.Accepted:
            self.weather_gw.zero_wind()
            print("前端控制器：已确认下发风速调零指令")

    def route_app1_click(self):
        """主界面按钮的智能跳转路由"""
        if self.is_collecting:
            # 如果已经在采集中，直接跳过标定和设置，切入监控面板
            self.enter_app(self.view.page_monitor)
        else:
            # 确保全局摄像头实例化，并注入给标定画布
            if self.global_camera is None:
                self.global_camera = GstPipelineManager(config=cfg, force_no_record=True)
            self.view.canvas.load_camera(self.global_camera)
            # 如果尚未运行，按照正常流程进入第一步预设
            self.enter_app(self.wizard_flow[0])
    
    def route_app2_click(self):
        """跳转至气象站校准页面"""
        self.enter_app(self.view.page_weather_calib)
        # TODO 进入时自动对齐时间

    def route_app3_click(self):
        """跳转至历史数据浏览页面"""
        self.enter_app(self.view.page_db_browser)
        self.handle_db_refresh() # 进入时自动拉取一次最新数据
    
    def toggle_headless(self):
        # 翻转状态
        self.is_headless = not getattr(self, 'is_headless', False)
        
        if self.is_headless:
            self.view.btn_headless.setText("显示画面")
            self.view.btn_headless.setStyleSheet(self.view.style_hollow_red)
            self.view.video_canvas.show_message("画面渲染已关闭，推理引擎仍在后端运行.")
        else:
            self.view.btn_headless.setText("隐藏画面")
            self.view.btn_headless.setStyleSheet(self.view.style_hollow_green)
            self.view.video_canvas.show_message("正在恢复渲染通道...")
        
        # 将状态同步给底层的引擎
        if getattr(self, 'worker', None) and getattr(self.worker, 'engine', None):
            self.worker.engine.headless_mode = self.is_headless

    def route_settings_click(self):
        """跳转至系统设置页面"""
        # 如果当前正在采集中，禁止进入设置
        if self.is_collecting:
            from ui.components.edge_dialog import EdgeMessageBox
            dialog = EdgeMessageBox(self.view, "提示", "任务运行中无法修改设置，请先结束采集。")
            dialog.exec_()
            return
            
        self.enter_app(self.view.page_settings)

    def enter_app(self, target_widget):
        """进入具体功能的槽函数（统一的页面跳转入口）"""
        current_widget = self.view.stack.currentWidget()
        
        # 硬件资源安全锁：只要离开标定页面，立即释放摄像头
        if current_widget == self.view.page_calibration and target_widget != self.view.page_calibration:
            self.view.canvas.stop_preview()
            print("前端控制器：已释放标定页面的视频资源")
        
        self.view.stack.setCurrentWidget(target_widget)
        self.update_nav_buttons()

    def return_to_home(self):
        """返回主界面"""
        # 返回主页时，如果不处于收集中，彻底释放摄像头资源
        if self.global_camera and not self.is_collecting:
            self.global_camera.stop()
            self.global_camera = None

        # 获取跳转前的当前页面
        current_widget = self.view.stack.currentWidget()
        
        # 执行原有的页面切换和资源释放逻辑
        self.enter_app(self.view.page_main_menu) 
        self.update_main_menu_btn_style()
        
        # 如果是从“系统设置”页面返回，并且当前配置为“本地视频”
        if current_widget == self.view.page_settings and not cfg.USE_CAMERA:
            # 延时一小下弹出，等待主界面动画平稳
            QTimer.singleShot(300, self.show_debug_mode_warning)

    def prev_page(self):
        """上一页触发逻辑"""
        current_page = self.view.stack.currentWidget()
        if current_page in self.wizard_flow:
            idx = self.wizard_flow.index(current_page)
            if idx > 0:
                prev_page_widget = self.wizard_flow[idx - 1]
                
                # 修复点：使用 enter_app 替代直接 setCurrentWidget
                self.enter_app(prev_page_widget)
                
                if prev_page_widget == self.view.page_calibration:
                    # 恢复标定页面时，重新分配全局摄像头
                    if self.global_camera is None:
                        self.global_camera = GstPipelineManager(cfg)
                    self.view.canvas.load_camera(self.global_camera)

    def next_page(self):
        """下一页触发逻辑"""
        current_page = self.view.stack.currentWidget()
        
        # --- 离开当前页面的处理逻辑 ---
        if current_page == self.view.page_pos_settings:
            self.save_physics_settings()

        # --- 状态机流转逻辑 ---
        if current_page in self.wizard_flow:
            idx = self.wizard_flow.index(current_page)
            if idx < len(self.wizard_flow) - 1:
                next_page_widget = self.wizard_flow[idx + 1]

                # 修复点：进入标定步骤时的处理，提前到路由跳转之前
                if next_page_widget == self.view.page_calibration:
                    if self.global_camera is None:
                        self.global_camera = GstPipelineManager(cfg)
                    self.view.canvas.load_camera(self.global_camera)

                # 修复点：使用 enter_app 统一接管路由，触发安全锁
                self.enter_app(next_page_widget)
                
                # --- 进入新页面的处理逻辑 ---
                if next_page_widget == self.view.page_monitor and not self.is_collecting:
                    self.start_monitoring()
                    self.is_collecting = True
                    
        self.update_nav_buttons()

    def show_debug_mode_warning(self):
        """弹出本地视频模式的物理失真声明"""
        dialog = EdgeMessageBox(
            self.view, 
            "⚙️ 调试模式已开启", 
            "检测到视频输入源为本地文件。算力波动可能导致视频画面处理无法贴合原始帧率。", 
            info_text="注意：此模式仅供系统连通性测试，运动学特征将产生失真，不能用于科学数据收集。",
            is_warning=False
        )
        dialog.exec_()

    def save_physics_settings(self):
        """将界面上的物理与环境参数同步到配置文件"""
        is_left_side = self.view.combo_wx_side.currentIndex() == 0
        dist_to_edge = self.view.wx_dist_to_edge
        
        # 结合上一步保存的 phys_w 计算绝对 x 坐标
        if is_left_side:
            final_wx_pos = 0.0 - dist_to_edge
        else:
            final_wx_pos = self.view.phys_w + dist_to_edge
            
        final_road_angle = float(self.view.slider_road_angle.value())
        
        # 更新到配置内存中 (AlignmentEngine 在 start_engine 时会去读取最新的 cfg)
        cfg.WEATHER_STATION_X_POS = final_wx_pos
        cfg.ROAD_DIRECTION_ANGLE = final_road_angle
        print(f"[Controller] 环境先验已更新: X坐标={final_wx_pos}m, 道路角度={final_road_angle}°")

    def update_nav_buttons(self):
        current_page = self.view.stack.currentWidget()
        
        # 1. 只有主页面隐藏整个底部导航栏
        is_home = (current_page == self.view.page_main_menu)
        self.view.nav_widget.setVisible(not is_home)
        if is_home:
            return
        
        # 2. 控制特定功能按钮
        self.view.btn_stop.setVisible(current_page == self.view.page_monitor and self.is_collecting)
        is_db_page = (current_page == self.view.page_db_browser)
        self.view.btn_delete_db.setVisible(is_db_page)
        self.view.btn_export_db.setVisible(is_db_page)

        # 控制无头模式按钮仅在监测页面可见
        is_monitor_page = (current_page == self.view.page_monitor)
        self.view.btn_headless.setVisible(is_monitor_page)
        
        # 如果在监测页且正在采集，确保它显示
        if is_monitor_page:
            self.view.btn_prev.setVisible(False)
            self.view.btn_next.setVisible(False)

        # 3. 控制“向导流”中的上一步/下一步
        if current_page in self.wizard_flow:
            idx = self.wizard_flow.index(current_page)
            
            # 第一页没有上一步
            self.view.btn_prev.setVisible(idx > 0)
            
            # 最后一页 (运行面板) 隐藏导航箭头，只能点“结束采集”或“返回主页”
            if current_page == self.view.page_monitor:
                self.view.btn_prev.setVisible(False)
                self.view.btn_next.setVisible(False)
            else:
                self.view.btn_next.setVisible(True)
                
                # 如果是正式启动前的最后一步
                if current_page == self.view.page_pos_settings:
                    self.view.btn_next.setText(" 开 始 ")
                    self.view.btn_next.setStyleSheet(self.view.style_hollow_green)
                else:
                    self.view.btn_next.setText("下一步 ▶")
                    self.view.btn_next.setStyleSheet(self.view.style_hollow_white)
        else:
            # 独立页面 (校准页、历史数据库浏览页)，只有“返回主页”按钮，隐藏前后导航
            self.view.btn_prev.setVisible(False)
            self.view.btn_next.setVisible(False)
    
    def update_main_menu_btn_style(self):
        """根据采集状态刷新主界面按钮颜色和文字"""
        if self.is_collecting:
            self.view.btn_app1.setText("多源数据采集 [运行中...]")
            self.view.btn_app1.setStyleSheet(self.view.style_hollow_green)
        else:
            self.view.btn_app1.setText("多源数据采集")
            self.view.btn_app1.setStyleSheet(self.view.style_hollow_white)
    
    # --- 任务启停控制 ---
    
    def stop_collection_trigger(self):
        """触发结束采集：弹出确认窗口"""
        dialog = EdgeMessageBox(self.view, "结束任务确认", "确定要结束当前的采集任务并关闭引擎吗？", "未保存的缓冲区数据可能会丢失。", is_warning=True)
        if dialog.exec_() == EdgeMessageBox.Accepted:
            self.stop_monitoring()
    
    # ui/controllers/main_controller.py

    def stop_monitoring(self):
        if not self.is_collecting: return
        
        print(">>> [Controller] 正在请求引擎停止...")
        self.view.btn_stop.setEnabled(False)
        self.view.btn_stop.setText("正在结算...")

        # 停止主引擎内部 while 循环
        if getattr(self, 'worker', None) and getattr(self.worker, 'engine', None):
            self.worker.engine.stop()

        # 1. 设置停止信号
        # 底层 monitor_engine.py 的 run() 循环检测到它后，会自己调用 cleanup()
        if hasattr(self, 'stop_event') and self.stop_event:
            self.stop_event.set() 

        # 2. 等待线程安全退出
        if self.worker:
            self.worker.quit()
            # 给引擎 4 秒钟时间完成最后的数据库落盘
            if not self.worker.wait(4000):
                print("[Warning] 引擎响应超时。")
        
        # 3. 线程退出后，再执行 UI 的收尾工作
        self.on_engine_finished()
    
    def handle_exit_request(self):
        """主界面退出系统拦截"""
        dialog = EdgeMessageBox(self.view, "退出系统确认", "确定要关闭应用程序吗？", "若有运行中的任务，系统将安全停机并保存数据。", is_warning=True)
        if dialog.exec_() == EdgeMessageBox.Accepted:
            self.view.close() # 触发 MainWindow closeEvent -> 执行 cleanup

    def cleanup(self):
        """处理整个应用的关闭回收 (由窗口 closeEvent 触发)"""
        if self.is_collecting:
            self.stop_monitoring()

        # 如果在未收集状态下退出，手动释放全局摄像头
        if self.global_camera and not self.is_collecting:
            self.global_camera.stop()
            self.global_camera = None
            print("前端控制器：已清理全局摄像头后台管线。")
        
        # 安全关闭气象网关 C++ 后台线程
        if hasattr(self, 'weather_gw') and self.weather_gw:
            self.weather_gw.stop()

        # 安全关闭数据库连接，触发数据落盘合并
        if hasattr(self, 'db') and self.db:
            self.db.close()
    
    # --- 界面渲染更新 ---
    def update_sys_board(self):
        """定期刷新左侧的系统状态看板"""
        # --- 刷新气象站看板 ---
        self._update_weather_boards()

        # 确保当前的界面是主菜单，如果在其他界面就不白费性能去读硬件状态了
        if self.view.stack.currentWidget() != self.view.page_main_menu:
            return
            
        # 确保我们在 view 中维护了 status_labels 字典
        if not hasattr(self.view, 'status_labels'):
            return
            
        labels = self.view.status_labels
        
        # 动态获取并填入数据
        if "系统时间" in labels:
            labels["系统时间"].setText(SysMonitor.get_system_time())
        if "SD卡存储" in labels:
            labels["SD卡存储"].setText(SysMonitor.get_sd_storage())
        if "SSD存储" in labels:
            labels["SSD存储"].setText(SysMonitor.get_ssd_storage())
        if "网络连接" in labels:
            labels["网络连接"].setText(SysMonitor.get_network_status())
        if "气象网关" in labels:
            labels["气象网关"].setText(SysMonitor.get_weather_gateway())
        if "CPU 温度" in labels:
            labels["CPU 温度"].setText(SysMonitor.get_cpu_temp())
        if "NPU 温度" in labels:
            # 可以在温度过高时让字变红预警
            temp_str = SysMonitor.get_npu_temp()
            labels["NPU 温度"].setText(temp_str)
            if "°C" in temp_str and float(temp_str.replace(" °C", "")) > 75.0:
                labels["NPU 温度"].setStyleSheet("color: #ff5555; border: none; font-weight: bold;")
            else:
                labels["NPU 温度"].setStyleSheet("color: #ffffff; border: none;")

    # 更新两个页面的气象标签
    def _update_weather_boards(self):
        if not self.weather_gw:
            return
            
        data = self.weather_gw.get_data()
        
        if data["isOnline"]:
            # 格式化数据字符串
            vals = {
                "温度": f"{data['temp']:.1f} °C",
                "湿度": f"{data['humidity']:.1f} %",
                "风速": f"{data['windSpeed']:.2f} m/s",
                "风向": f"{data['windDir']} °",
                "PM2.5": f"{data['pm25']} μg/m³",
                "PM10": f"{data['pm10']} μg/m³"
            }
            # 将时间戳转换为可读时间
            dt_str = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            ts_text = f"远端系统时间戳: {dt_str}"
        else:
            # 离线占位符
            vals = {k: "--" for k in ["温度", "湿度", "风速", "风向", "PM2.5", "PM10"]}
            ts_text = "远端系统时间戳: 离线 / 等待同步..."

        # 同时刷新 Monitor 页 (Index 3) 和 校准页 (Index 4) 的字典
        if hasattr(self.view, 'weather_monitor_labels'):
            for key, lbl in self.view.weather_monitor_labels.items():
                if key in vals: lbl.setText(vals[key])
                
        if hasattr(self.view, 'weather_calib_labels'):
            for key, lbl in self.view.weather_calib_labels.items():
                if key in vals: lbl.setText(vals[key])
                
        # 更新校准页的时间戳标签
        if hasattr(self.view, 'lbl_calib_timestamp'):
            self.view.lbl_calib_timestamp.setText(ts_text)
            if data["isOnline"]:
                self.view.lbl_calib_timestamp.setStyleSheet("color: #00e676; border: none;") # 上线变绿
            else:
                self.view.lbl_calib_timestamp.setStyleSheet("color: #f39c12; border: none;") # 离线变黄

    def handle_db_refresh(self):
        """刷新下拉菜单中的采集任务列表"""
        self.view.btn_refresh_db.setText(" 读取中... ")
        self.view.btn_refresh_db.repaint()
        try:
            db = DatabaseManager()
            sessions = db.fetch_all_sessions()
            
            # 记录当前选中的选项，防止刷新后跳走
            current_session = self.view.session_combo.currentData()
            
            # 屏蔽信号，防止在清空和添加下拉项时频繁触发 currentIndexChanged
            self.view.session_combo.blockSignals(True)
            self.view.session_combo.clear()
            
            if not sessions:
                self.view.session_combo.addItem("当前暂无采集任务记录", None)
            else:
                for sess in sessions:
                    sid, st, desc = sess
                    dt_str = datetime.fromtimestamp(st).strftime('%m-%d %H:%M')
                    display_text = f"[{dt_str}] {desc}"
                    self.view.session_combo.addItem(display_text, sid)
                    
            # 尝试恢复之前的选中状态
            idx = self.view.session_combo.findData(current_session)
            if idx >= 0:
                self.view.session_combo.setCurrentIndex(idx)
            elif sessions:
                self.view.session_combo.setCurrentIndex(0)
                
            self.view.session_combo.blockSignals(False)
            
            # 触发当前选中任务的表格刷新
            self.update_db_table()
            db.close()
        except Exception as e:
            print(f"下拉菜单刷新失败: {e}")
        finally:
            self.view.btn_refresh_db.setText(" 刷新数据 ")

    def update_db_table(self):
        """拉取当前选中 Session 的数据并渲染到表格"""
        session_id = self.view.session_combo.currentData()
        table = self.view.db_table
        table.setRowCount(0) # 清空旧数据
        
        if not session_id:
            return
            
        try:
            db = DatabaseManager()
            records = db.fetch_macro_records_by_session(session_id, limit=50)
            db.close()

            for row_idx, row_data in enumerate(records):
                table.insertRow(row_idx)
                for col_idx, col_value in enumerate(row_data):
                    val_str = str(col_value) if col_value is not None else "--"
                    
                    if col_idx in [3, 4] and col_value is not None:
                        try:
                            dt = datetime.fromtimestamp(float(col_value))
                            val_str = dt.strftime('%H:%M:%S.%f')[:-4]
                        except Exception: pass
                            
                    if col_idx == 6 and col_value is not None:
                        try:
                            opmodes_list = json.loads(col_value)
                            val_str = ", ".join(opmodes_list)
                        except Exception: pass

                    item = QTableWidgetItem(val_str)
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(row_idx, col_idx, item)
        except Exception as e:
            print(f"数据表格刷新失败: {e}")

    def show_batch_delete_dialog(self):
        """弹出基于 Session 的删除数据对话框"""

        # 空数据库拦截
        if self.view.session_combo.currentData() is None:
            # 默认 is_warning=False，会弹出一个带有白色/绿色确认按钮的常规提示框
            dialog = EdgeMessageBox(
                self.view, 
                "无历史数据", 
                "当前边缘节点数据库为空。", 
                info_text="没有任何可清理的任务数据。"
            )
            dialog.exec_()
            return

        # 数据库不为空才进行后续步骤
        dialog = EdgeAnimatedDialog(self.view, target_height=260, is_warning=True)
        
        layout = QVBoxLayout(dialog.panel)
        layout.setContentsMargins(40, 30, 40, 30)

        title = QLabel("清理历史数据")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #ffffff; border: none;")
        layout.addWidget(title)
        layout.addSpacing(15)
        
        current_session_id = self.view.session_combo.currentData()
        current_session_text = self.view.session_combo.currentText()
        
        # --- 模式选择 ---
        radio_session = QRadioButton(f"删除当前选中的任务: {current_session_text}")
        radio_session.setStyleSheet("color: #dddddd; font-size: 16px;")
        radio_session.setChecked(True)
        if not current_session_id: radio_session.setEnabled(False)
            
        radio_all = QRadioButton("清空【所有】历史任务数据")
        radio_all.setStyleSheet("color: #ff4d4f; font-size: 16px; font-weight: bold;")
        
        layout.addWidget(radio_session)
        layout.addWidget(radio_all)
        layout.addStretch()
        
        # --- 底部确认/取消按钮 ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_cancel = QPushButton("取消")
        btn_cancel.setFixedSize(120, 45)
        btn_cancel.setFont(QFont("Arial", 14, QFont.Bold))
        btn_cancel.setStyleSheet("background-color: transparent; border: 2px solid #777; color: #fff; border-radius: 5px;")
        btn_cancel.clicked.connect(lambda: dialog.close_with_anim(EdgeAnimatedDialog.Rejected))
        
        btn_confirm = QPushButton("确认删除")
        btn_confirm.setFixedSize(120, 45)
        btn_confirm.setFont(QFont("Arial", 14, QFont.Bold))
        btn_confirm.setStyleSheet("background-color: transparent; border: 2px solid #ff4d4f; color: #ff4d4f; border-radius: 5px;")
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(btn_confirm)
        layout.addLayout(btn_layout)
        
        # --- 第二级危险确认逻辑 ---
        def execute_delete():
            msg = "您即将【清空所有的历史数据记录】。\n此操作不可逆，确定继续吗？" if radio_all.isChecked() else "您即将删除当前选中的任务数据。\n此操作不可逆，确定继续吗？"
            confirm_dialog = EdgeMessageBox(dialog, "⚠️ 最终删除确认", msg, is_warning=True)
            if confirm_dialog.exec_() == EdgeMessageBox.Accepted:
                db = DatabaseManager()
                success = False
                if radio_all.isChecked():
                    success = db.delete_all_data()
                elif current_session_id:
                    success = db.delete_session(current_session_id)
                db.close()
                
                if success:
                    self.handle_db_refresh() 
                dialog.close_with_anim(EdgeAnimatedDialog.Accepted)
                
        btn_confirm.clicked.connect(execute_delete)
        dialog.exec_()
    
    def handle_export_db_data(self):
        """处理当前选中 Session 的数据库导出请求"""
        # 1. 获取当前下拉菜单选中的 Session ID
        current_session_id = self.view.session_combo.currentData()
        
        if not current_session_id:
            EdgeMessageBox(self.view, "无数据可导出", "当前没有可供导出的任务记录。").exec_()
            return

        # 2. U 盘检测 (复用现有的安全检索逻辑)
        usbs = StorageManager.get_available_usbs()
        if not usbs:
            dialog = EdgeMessageBox(
                self.view, 
                "未检测到外部存储", 
                "请将 U 盘插入设备的 USB 接口。", 
                info_text="系统仅支持导出至挂载于 /media 目录下的设备。",
                is_warning=True
            )
            dialog.exec_()
            return
            
        target_usb_path = usbs[0]

        # 3. 执行导出
        self.view.btn_export_db.setText(" 正在导出... ")
        self.view.btn_export_db.setEnabled(False)
        self.view.btn_export_db.repaint()
        
        try:
            # 调用底层方法
            target_dir, files = StorageManager.export_data_to_usb(
                current_session_id, target_usb_path, self.db
            )
            
            file_names_str = "\n".join(files)
            EdgeMessageBox(
                self.view, 
                "导出成功", 
                f"已成功将任务数据导出至 U 盘。",
                info_text=f"保存目录:\n{target_dir.name}\n\n包含文件:\n{file_names_str}"
            ).exec_()
            
        except Exception as e:
            EdgeMessageBox(self.view, "导出失败", f"数据转换或拷贝时发生错误: {e}", is_warning=True).exec_()
        finally:
            self.view.btn_export_db.setText("导出数据")
            self.view.btn_export_db.setEnabled(True)

    def update_timer_tasks(self):
        """总控定时器：分配 UI 刷新任务"""
        if not self.worker or not getattr(self.worker, 'engine', None): 
            return
        
        self._update_thermal_view()  # 1. 实时刷新热成像
        self._update_dashboard()     # 2. 刷新车辆抽样看板

    def _reload_global_camera(self):
        """根据新配置销毁并重载摄像头"""
        if self.global_camera:
            self.global_camera.stop()
            self.global_camera = None
            
        # 如果当前正停留在标定页，立马重载它（通常不会发生，因为设置页独立）
        if self.view.stack.currentWidget() == self.view.page_calibration:
            self.global_camera = GstPipelineManager(cfg)
            self.view.canvas.load_camera(self.global_camera)

    def handle_browse_local_video(self):
        """处理点击浏览本地视频文件的逻辑"""
        self.view.radio_source_local.setChecked(True)
        
        # 1. 明确起始目录为 StorageManager 规范的测试目录
        init_dir = str(StorageManager.TEST_DIR)
        
        # 呼出系统文件选择对话框 (建议传入 self.view 作为父组件居中显示)
        file_path, _ = QFileDialog.getOpenFileName(
            self.view,
            "选择本地测试视频文件",
            init_dir, # 限制起始目录
            "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*.*)"
        )
        
        if file_path:
            # 2. 安全沙盒拦截：不允许选择 data/test_videos 以外的文件
            if not file_path.startswith(str(StorageManager.DATA_ROOT)):
                dialog = EdgeMessageBox(
                    self.view, 
                    "⚠️ 路径非法", 
                    "为保障容器隔离安全，只能选择应用数据目录下的视频文件。", 
                    info_text=f"允许的根目录: {StorageManager.DATA_ROOT}",
                    is_warning=True
                )
                dialog.exec_()
                return

            # 3. 更新 UI 显示与配置
            self.view.lbl_local_path.setText(file_path)
            cfg.update_source_settings(file_path, use_camera=False)
            self._reload_global_camera() 
            print(f"前端控制器：已更新视频输入路径为 {file_path}，并保存至配置文件")

    def handle_detect_camera(self):
        """搜索物理设备并生成 GStreamer 管道"""
        self.view.radio_source_camera.setChecked(True)
        
        camera_path = "/dev/video0"
        
        # 针对树莓派，检查视频设备节点是否存在
        if os.path.exists(camera_path):
            # 获取针对 RPi 硬件加速优化的管道字符串
            pipeline = gst.get_rpi_camera_pipeline(cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT, cfg.FPS)
            
            # 动态读取 Linux 底层硬件传感器名称
            camera_name = "未知型号摄像头"
            sysfs_path = "/sys/class/video4linux/video0/name"
            if os.path.exists(sysfs_path):
                try:
                    with open(sysfs_path, 'r', encoding='utf-8') as f:
                        camera_name = f.read().strip()
                except Exception as e:
                    print(f"读取摄像头硬件名称失败: {e}")
            
            # 拼接具体路径和动态获取到的型号信息
            display_text = f"已接入: {camera_name} ({camera_path})"
            self.view.lbl_camera_info.setText(display_text)
            
            cfg.update_source_settings(pipeline, use_camera=True)
            self._reload_global_camera() # 销毁旧实例
            print(f"控制器：已切换至物理摄像头流: {pipeline}")
        else:
            self.view.lbl_camera_info.setText(f"未发现摄像头设备 ({camera_path})")

    def handle_source_type_changed(self):
        """当用户手动切换视频源单选框时更新配置"""
        is_camera = self.view.radio_source_camera.isChecked()
        
        # 仅在摄像头源时激活录制选项页签的内容
        self.view.tab_record_settings.setEnabled(is_camera)

        if self.view.radio_source_local.isChecked():
            # 切换到本地模式
            current_path = self.view.lbl_local_path.text()
            cfg.update_source_settings(current_path, use_camera=False)
            self._reload_global_camera()
        
        elif self.view.radio_source_camera.isChecked():
            # 只要当前提示标签显示为“已接入”，就自动切换配置
            # 这样用户在已经看到“已接入...”的情况下，点选单选框即可生效，无需再按“检测”
            info_text = self.view.lbl_camera_info.text()
            if "已接入" in info_text:
                pipeline = gst.get_rpi_camera_pipeline(cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT, cfg.FPS)
                cfg.update_source_settings(pipeline, use_camera=True)
                self._reload_global_camera() # 【新增】
                print(f"控制器：已根据现有状态自动切换至摄像头: {pipeline}")

    def handle_run_mode_changed(self):
        """处理工作模式切换"""
        if self.view.radio_mode_inference.isChecked():
            cfg.update_run_mode('inference')
        else:
            cfg.update_run_mode('collection')

    def handle_record_switch_toggled(self, checked):
        """处理录制开关的 UI 状态变化"""
        if checked:
            self.view.btn_record_switch.setText("已开启")
            self.view.btn_record_switch.setStyleSheet(self.view.style_hollow_green)
        else:
            self.view.btn_record_switch.setText("已关闭")
            self.view.btn_record_switch.setStyleSheet(self.view.style_hollow_white)
        self.save_record_settings() # 状态变化即保存

    def handle_export_videos(self):
        """处理将视频导出至外部 U 盘的请求"""
        # 1. 硬件检测：检查是否有挂载的 U 盘
        usbs = StorageManager.get_available_usbs()
        if not usbs:
            dialog = EdgeMessageBox(
                self.view, 
                "未检测到外部存储", 
                "请将 U 盘插入树莓派的 USB 接口。", 
                info_text="系统仅支持导出至挂载于 /media 目录下的设备。",
                is_warning=True
            )
            dialog.exec_()
            return
            
        target_usb_path = usbs[0] # 默认使用检测到的第一个 U 盘

        # 2. 数据检测：获取所有含有录像的任务
        session_videos = StorageManager.get_session_videos()
        if not session_videos:
            dialog = EdgeMessageBox(self.view, "无视频数据", "当前本地存储中没有可导出的视频文件。")
            dialog.exec_()
            return

        # 3. 构建给弹窗用的精简数据映射: {session_id: 视频数量}
        session_data_map = {sid: len(files) for sid, files in session_videos.items()}

        # 4. 呼出多选弹窗
        export_dialog = EdgeExportDialog(self.view, session_data_map)
        if export_dialog.exec_() == EdgeMessageBox.Accepted:
            selected_sids = export_dialog.selected_sessions
            if not selected_sids:
                return # 啥都没选
                
            self.view.btn_export_videos.setText(" 正在导出... ")
            self.view.btn_export_videos.setEnabled(False)
            self.view.btn_export_videos.repaint() # 强制立刻刷新 UI
            
            # 5. 执行文件拷贝
            # 注意：实际生产中大量大文件拷贝建议放入 QThread 防止 UI 假死。
            # 这里为了保持逻辑紧凑，先在主线程执行同步拷贝
            try:
                exported_count = 0
                for sid in selected_sids:
                    for video_name in session_videos[sid]:
                        # 在这里追加 session_id=sid 参数
                        StorageManager.export_to_usb(video_name, target_usb_path, session_id=sid)
                        exported_count += 1
                        
                EdgeMessageBox(
                    self.view, 
                    "导出成功", 
                    f"已成功将 {exported_count} 个视频文件导出至 U 盘。",
                    info_text=f"目标路径: {target_usb_path.name}/{selected_sids[0]}_recorded_videos 等"
                ).exec_()
            except Exception as e:
                EdgeMessageBox(self.view, "导出失败", f"文件拷贝过程中发生错误: {e}", is_warning=True).exec_()
            finally:
                self.view.btn_export_videos.setText(" 导出视频至外部 U 盘 ")
                self.view.btn_export_videos.setEnabled(True)

    def save_record_settings(self):
        """将当前的录制 UI 状态同步到配置文件"""
        enable = self.view.btn_record_switch.isChecked()
        seg_str = self.view.combo_segment_time.currentText()
        seg_min = int(seg_str.split()[0]) # 提取出纯数字 5, 10, 15...
        path = str(StorageManager.REC_DIR)
        cfg.update_record_settings(enable, seg_min, path)

    def check_and_cleanup_disk_space(self):
        """后台守护：检查磁盘空间，不足时删除最早的切片视频"""
        if not cfg.ENABLE_RECORD or not cfg.USE_CAMERA:
            return
            
        path = cfg.RECORD_SAVE_PATH
        if not os.path.exists(path):
            return

        import shutil
        # 设定安全红线：低于 2GB 时触发清理 (后续可根据需求调整，或可配置化)
        MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024 
        
        try:
            usage = shutil.disk_usage(path)
            if usage.free < MIN_FREE_BYTES:
                # 寻找该目录下所有的 .mp4 文件
                files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mp4')]
                if not files: return
                
                # 按照文件创建时间排序，找到最旧的
                oldest_file = min(files, key=os.path.getctime)
                os.remove(oldest_file)
                print(f"[磁盘管理] 存储空间低于阈值，已自动清理最旧的切片: {oldest_file}")
        except Exception as e:
            print(f"[磁盘管理] 清理监控异常: {e}")

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
            
            self.view.lbl_thermal_min.setText(f"最低温度: {t_min:.1f} °C")
            self.view.lbl_thermal_max.setText(f"最高温度: {t_max:.1f} °C")
            self.view.lbl_thermal_center.setText(f"中心温度: {t_center:.1f} °C")
            
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
            self.view.thermal_label.setPixmap(QPixmap.fromImage(qimg))
    
    def _update_dashboard(self):
        """Dashboard 数据抽样更新逻辑 (离场后结算展示)"""
        # --- 会话安全检查 ---
        # 如果当前没有 session_id，说明任务尚未通过 start_monitoring 正式启动
        # 此时不应进行任何数据库查询或 UI 刷新
        if not getattr(self, 'current_session_id', None):
            return

        # --- 线程与引擎安全检查 ---
        # 使用“短路逻辑”防御：如果 worker 是 None，或者其引用的 engine 尚未挂载，立刻退出
        if not self.worker or not getattr(self.worker, 'engine', None): 
            return
        
        engine = self.worker.engine

        # --- 数据库读取与表格刷新 ---
        # 确保数据库可用，并实时刷新下方的大表格
        if self.db:
            try:
                # 拿着当前的 session_id 去拉取最新的 50 条结算记录
                records = self.db.fetch_macro_records_by_session(self.current_session_id, limit=50)
                # 调用同步方法更新表格 UI
                self._sync_table_records(records)
            except Exception as e:
                print(f"[UI] 刷新结算表格失败: {e}")

        # --- 单车抽样数据检查 ---
        # 检查引擎是否产出了最新结算完毕的离场数据，使用 getattr 更加安全
        latest_data = getattr(engine, 'latest_exit_record', None)
        if not latest_data:
            return
            
        tid = latest_data.get('tid')

        # 如果这辆车已经在 Dashboard 上展示过了，就不重复刷新，避免 UI 闪烁
        if getattr(self, 'sampled_tid', None) == tid:
            return
            
        # --- 更新单车详细数据面板 ---
        self.sampled_tid = tid
        record = latest_data.get('record', {})
        type_str = latest_data.get('type_str')
        
        # 读取引擎层结算好的投票结果，保持与落盘数据 100% 一致
        plate_color = record.get('final_plate_color', 'Unknown')
            
        # 提取经过 S-G 非因果滤波处理过的高质量速度、加速度曲线
        trajectory = record.get('trajectory', [])
        speeds = [p['speed'] for p in trajectory if 'speed' in p]
        accels = [p['accel'] for p in trajectory if 'accel' in p]
        
        # 更新 UI 组件
        self.view.lbl_dash_id.setText(f"目标 ID: #{tid}")

        type_zh_map = {
            "LDV-Gasoline": "轻型燃油车 (LDV)",
            "LDV-Electric": "轻型新能源 (LDV)",
            "HDV-Diesel": "重型柴油车 (HDV)",
            "HDV-Electric": "重型新能源 (HDV)"
        }
        display_type = type_zh_map.get(type_str, type_str) if type_str else "未知"
        self.view.lbl_dash_type.setText(f"车型: {display_type}")

        color_map = {'blue': '蓝色', 'green': '绿色', 'yellow': '黄色', 'white': '白色', 'black': '黑色', 'Unknown': '未知'}
        zh_color = color_map.get(plate_color, plate_color)
        self.view.lbl_dash_plate.setText(f"车牌颜色: {zh_color}")
        self.view.lbl_dash_dist.setText(f"行驶距离: {record.get('total_distance_m', 0.0):.1f} m")
        
        self.view.curve_widget.update_curve(speeds, accels)

    def _sync_table_records(self, records):
        """
        @brief 将数据库查询出的最新车辆记录同步渲染到 UI 的结算表格中
        @param records: List[tuple] 从 SQLite 拉取的记录列表
        """
        if not hasattr(self.view, 'db_table'):
            return

        # 1. 动态设置表格总行数
        self.view.db_table.setRowCount(len(records))
        
        # 车型与状态的中文化映射字典
        type_zh_map = {
            "LDV-Gasoline": "轻型燃油",
            "LDV-Electric": "轻型新能源",
            "HDV-Diesel": "重型柴油",
            "HDV-Electric": "重型新能源"
        }
        status_map = {
            "Completed": "正常离场", 
            "Timeout": "超时清理", 
            "Border": "边缘截断"
        }

        # 2. 遍历数据并逐行填充
        for row_idx, record in enumerate(records):
            # record 对应 SQL: tracker_id, vehicle_type, energy_type, entry_time, exit_time, average_speed, dominant_opmodes, settlement_status
            tid, v_type, e_type, en_time, ex_time, speed, opmodes, status = record
            
            # 格式化时间戳为可读时间 (时:分:秒)
            en_str = datetime.fromtimestamp(en_time).strftime('%H:%M:%S') if en_time else "--:--"
            ex_str = datetime.fromtimestamp(ex_time).strftime('%H:%M:%S') if ex_time else "--:--"
            
            # 格式化车型与状态
            display_type = type_zh_map.get(v_type, v_type) if v_type else "未知"
            display_status = status_map.get(status, status) if status else "未知"

            # 创建表格 Item (注意字段顺序需要与 main_window.py 中的表头定义一致)
            items = [
                QTableWidgetItem(f"#{tid}"),                 # 列 0: 目标 ID
                QTableWidgetItem(display_type),              # 列 1: 车型
                QTableWidgetItem(e_type if e_type else "-"), # 列 2: 能源类型
                QTableWidgetItem(en_str),                    # 列 3: 入场时间
                QTableWidgetItem(ex_str),                    # 列 4: 离场时间
                QTableWidgetItem(f"{speed:.1f}" if speed else "0.0"), # 列 5: 均速
                QTableWidgetItem(str(opmodes).strip('[]')),  # 列 6: 主导工况 (剥除JSON数组符号)
                QTableWidgetItem(display_status)             # 列 7: 结算状态
            ]
            
            # 3. 统一设置居中对齐，并塞入 Table
            for col_idx, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                self.view.db_table.setItem(row_idx, col_idx, item)
