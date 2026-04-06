import time
import json
import cv2
import numpy as np
import infra.config.loader as cfg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QMessageBox, QTableWidgetItem, QDialog, 
                             QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QRadioButton)
from PyQt5.QtGui import QImage, QPixmap, QFont
from datetime import datetime
from ui.workers.engine_worker import EngineWorker
from infra.sys.sys_monitor import SysMonitor
from infra.store.sqlite_manager import DatabaseManager
from perception.sensor.weather_station import WeatherGateway
from domain.physics.alignment_engine import DelayedAlignmentEngine

class MainController:
    """Controller 层：负责状态管理、页面路由、信号绑定与定时更新"""
    def __init__(self, view):
        self.view = view
        self.is_collecting = False
        self.sampled_tid = None
        self.worker = None

        # 定义显式的页面向导流
        # 只要在这里修改顺序，整个向导流就会自动适应，不需要改其他逻辑
        self.wizard_flow = [
            self.view.page_calibration,       # 步骤 1: 视觉标定
            self.view.page_size_settings,     # 步骤 2: 道路ROI尺寸设置
            self.view.page_pos_settings,      # 步骤 3: 仪器与道路方位设置
            self.view.page_monitor            # 运行: 监控面板
        ]
        
        self.dash_timer = QTimer(self.view)
        self.dash_timer.timeout.connect(self.update_timer_tasks)

        # 实例化并启动气象网关
        try:
            self.weather_gw = WeatherGateway()
            self.weather_gw.start()
        except Exception as e:
            print(f"气象驱动加载失败: {e}")
            self.weather_gw = None

        self.bind_signals()
        self.view.close_callback = self.cleanup  # 注入关闭事件钩子
        self.update_nav_buttons()
        self.update_main_menu_btn_style()

        # 用于慢速轮询硬件状态的定时器 (1Hz)
        self.sys_timer = QTimer(self.view)
        self.sys_timer.timeout.connect(self.update_sys_board)
        self.sys_timer.start(1000) # 每 1000 毫秒刷新一次系统看板

    def bind_signals(self):
        """将视图组件的事件绑定到控制器的逻辑上"""
        self.view.btn_home.clicked.connect(self.return_to_home)
        self.view.btn_stop.clicked.connect(self.stop_collection_trigger)
        self.view.btn_prev.clicked.connect(self.prev_page)
        self.view.btn_next.clicked.connect(self.next_page)
        self.view.btn_app1.clicked.connect(self.route_app1_click)
        self.view.btn_app2.clicked.connect(self.route_app2_click)
        self.view.btn_app3.clicked.connect(self.route_app3_click)
        self.view.btn_exit.clicked.connect(self.view.close)

        # 绑定气象站的校准按钮事件
        self.view.btn_sync_clock.clicked.connect(self.handle_sync_clock)
        self.view.btn_zero_wind.clicked.connect(self.handle_zero_wind)

        # 数据表单的按钮/列表绑定
        self.view.btn_refresh_db.clicked.connect(self.handle_db_refresh)
        self.view.btn_delete_db.clicked.connect(self.show_batch_delete_dialog)
        self.view.session_combo.currentIndexChanged.connect(self.update_db_table)
    
    def handle_sync_clock(self):
        if not self.weather_gw:
            return
            
        # 创建确认弹窗
        msg_box = QMessageBox(self.view)
        msg_box.setWindowTitle("时钟同步校准")
        msg_box.setText("确认与远端系统同步时间吗？")
        msg_box.setIcon(QMessageBox.Question)
        
        # 自定义按钮中文文本
        yes_btn = msg_box.addButton("确认", QMessageBox.YesRole)
        no_btn = msg_box.addButton("取消", QMessageBox.NoRole)
        
        msg_box.exec_()
        
        # 仅当用户点击确认时才下发指令
        if msg_box.clickedButton() == yes_btn:
            self.weather_gw.sync_time()
            print("前端控制器：已确认下发时钟同步指令")
            
    def handle_zero_wind(self):
        if not self.weather_gw:
            return
            
        # 创建确认弹窗
        msg_box = QMessageBox(self.view)
        msg_box.setWindowTitle("风速调零校准")
        msg_box.setText("确认执行风速调零吗？\n请确保传感器处于无风环境，等待10秒后完成调零。")
        msg_box.setIcon(QMessageBox.Warning) # 调零属于敏感操作，使用黄色警告图标
        
        # 自定义按钮中文文本
        yes_btn = msg_box.addButton("确认", QMessageBox.YesRole)
        no_btn = msg_box.addButton("取消", QMessageBox.NoRole)
        
        msg_box.exec_()
        
        # 仅当用户点击确认时才下发指令
        if msg_box.clickedButton() == yes_btn:
            self.weather_gw.zero_wind()
            print("前端控制器：已确认下发风速调零指令")

    def route_app1_click(self):
        """主界面按钮的智能跳转路由"""
        if self.is_collecting:
            # 如果已经在采集中，直接跳过标定和设置，切入监控面板
            self.enter_app(self.view.page_monitor)
        else:
            # 如果尚未运行，按照正常流程进入第一步标定环节
            self.enter_app(self.view.page_weather_calib)
    
    def route_app2_click(self):
        """跳转至气象站校准页面"""
        self.enter_app(self.view.page_weather_calib)
        # TODO 进入时自动对齐时间

    def route_app3_click(self):
        """跳转至历史数据浏览页面"""
        self.enter_app(self.view.page_db_browser)
        self.handle_db_refresh() # 进入时自动拉取一次最新数据

    def enter_app(self, target_widget):
        """进入具体功能的槽函数"""
        self.view.stack.setCurrentIndex(target_widget)
        self.update_nav_buttons()

    def return_to_home(self):
        """返回主界面"""
        self.view.stack.setCurrentIndex(self.view.page_main_menu)
        self.update_nav_buttons()
        self.update_main_menu_btn_style()

    def prev_page(self):
        """上一页触发逻辑"""
        current_page = self.view.stack.currentWidget()
        if current_page in self.wizard_flow:
            idx = self.wizard_flow.index(current_page)
            if idx > 0: # 只要不是第一步，就可以回退
                self.view.stack.setCurrentWidget(self.wizard_flow[idx - 1])
        self.update_nav_buttons()

    def next_page(self):
        """下一页触发逻辑"""
        current_page = self.view.stack.currentWidget()
        
        # --- 离开当前页面的处理逻辑 ---
        if current_page == self.view.page_pos_settings:
            self.save_physics_settings() # 保存物理参数设置

        # --- 状态机流转逻辑 ---
        if current_page in self.wizard_flow:
            idx = self.wizard_flow.index(current_page)
            if idx < len(self.wizard_flow) - 1:
                next_page = self.wizard_flow[idx + 1]
                self.view.stack.setCurrentWidget(next_page)
                
                # --- 进入新页面的处理逻辑 ---
                if next_page == self.view.page_monitor and not self.is_collecting:
                    self.start_engine()
                    self.dash_timer.start(100)  # 10Hz 轮询更新 Dashboard
                    self.is_collecting = True
                    
        self.update_nav_buttons()

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
        self.view.btn_delete_db.setVisible(current_page == self.view.page_db_browser)

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
                if current_page == self.view.page_physics_settings:
                    self.view.btn_next.setText(" 开 始 ")
                    self.view.btn_next.setStyleSheet("background-color: #4CAF50; color: white;")
                else:
                    self.view.btn_next.setText("下一步 ▶")
                    self.view.btn_next.setStyleSheet("")
        else:
            # 独立页面 (校准页、历史数据库浏览页)，只有“返回主页”按钮，隐藏前后导航
            self.view.btn_prev.setVisible(False)
            self.view.btn_next.setVisible(False)
    
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
            self.view.btn_app1.setText("多源数据采集 (运行中...)")
        else:
            style = """
                QPushButton {
                    background-color: #2962ff; color: white; border: none; border-radius: 8px;
                    padding: 15px; text-align: left; padding-left: 20px;
                }
                QPushButton:hover { background-color: #0039cb; }
            """
            self.view.btn_app1.setText("多源数据采集")
        
        self.view.btn_app1.setStyleSheet(style)
    
    # --- 任务启停控制 ---
    def start_engine(self):
        # 提取的是反算好的原生 1080p 真实坐标
        source_points = self.view.canvas.get_real_points()
        
        # 启动后台引擎线程
        self.worker = EngineWorker(source_points, self.view.phys_w, self.view.phys_h, weather_station=self.weather_gw)
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.start()

        # 启动延迟对齐引擎
        self.align_engine = DelayedAlignmentEngine(cfg._cfg, cfg.DB_PATH)
        self.align_timer = QTimer(self.view)
        freq = cfg._cfg["time_windows"].get("db_align_frequency_hz", 1.0)
        self.align_timer.timeout.connect(self._run_alignment_step)
        # 根据配置文件频率设置定时器（1.0Hz -> 1000ms）
        self.align_timer.start(int(1000 / freq))
    
    def stop_collection_trigger(self):
        """触发结束采集：弹出确认窗口"""
        msg_box = QMessageBox(self.view)
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
        if hasattr(self, 'align_timer'): self.align_timer.stop()
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.worker.wait(1000)
        
        self.is_collecting = False
        self.update_main_menu_btn_style()
        self.return_to_home()
    
    def cleanup(self):
        """处理整个应用的关闭回收"""
        if self.is_collecting:
            self.final_stop_process()
        
        # 安全关闭气象网关 C++ 后台线程，防止内存或串口泄漏
        if self.weather_gw:
            self.weather_gw.stop()
    
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
        if "边缘存储" in labels:
            labels["边缘存储"].setText(SysMonitor.get_edge_storage())
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
        dialog = QDialog(self.view)
        dialog.setWindowTitle("清理历史数据")
        dialog.setFixedSize(480, 220) 
        dialog.setStyleSheet("""
            QDialog { background-color: #1a1d2d; border: 2px solid #2d324f; }
            QLabel { color: white; font-size: 16px; }
            QRadioButton { color: white; font-size: 18px; font-weight: bold; }
            QRadioButton::indicator { width: 20px; height: 20px; }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(30, 30, 30, 20)
        
        # 获取当前选中的会话信息
        current_session_id = self.view.session_combo.currentData()
        current_session_text = self.view.session_combo.currentText()
        
        # --- 模式选择 ---
        radio_session = QRadioButton(f"删除当前选中的任务:\n({current_session_text})")
        radio_session.setChecked(True)
        if not current_session_id:
            radio_session.setEnabled(False) # 如果没有数据，禁用此选项
            
        radio_all = QRadioButton("清空【所有】历史任务数据")
        
        layout.addWidget(radio_session)
        layout.addSpacing(15)
        layout.addWidget(radio_all)
        layout.addStretch()
        
        # --- 底部确认/取消按钮 ---
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("取消")
        btn_cancel.setStyleSheet("background-color: #555; color: white; font-size: 16px; padding: 12px; border-radius: 5px;")
        
        btn_confirm = QPushButton("确认")
        btn_confirm.setStyleSheet("background-color: #d50000; color: white; font-weight: bold; font-size: 16px; padding: 12px; border-radius: 5px;")
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_confirm)
        layout.addLayout(btn_layout)
        
        btn_cancel.clicked.connect(dialog.reject)
        
        # --- 第二级危险确认逻辑 ---
        def execute_delete():
            msg_box = QMessageBox(dialog)
            msg_box.setWindowTitle("⚠️ 删除操作确认")
            msg_box.setIcon(QMessageBox.Critical)
            if radio_all.isChecked():
                msg_box.setText("您即将【清空所有的历史数据记录】。\n此操作执行后数据将无法找回，确定继续吗？")
            else:
                msg_box.setText("您即将删除当前选中的任务数据。\n此操作执行后无法找回，确定继续吗？")
                
            yes_btn = msg_box.addButton("确认删除", QMessageBox.YesRole)
            no_btn = msg_box.addButton("放弃删除", QMessageBox.NoRole)
            msg_box.exec_()
            
            if msg_box.clickedButton() == yes_btn:
                db = DatabaseManager()
                success = False
                if radio_all.isChecked():
                    success = db.delete_all_data()
                elif current_session_id:
                    success = db.delete_session(current_session_id)
                db.close()
                
                if success:
                    # 删除成功后，彻底重载下拉框和表格
                    self.handle_db_refresh() 
                dialog.accept()
                
        btn_confirm.clicked.connect(execute_delete)
        dialog.exec_()

    def update_video_frame(self, rgb_img):
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 利用 Qt 的机制自适应当前 Label 尺寸，自动补齐黑边并保持等比缩放
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.view.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.view.video_label.setPixmap(scaled_pixmap)

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
            
        # 捕获到了新的离场车辆
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

    def _run_alignment_step(self):
        """定时触发的对齐任务钩子"""
        if not self.is_collecting:
            return
            
        # 检查底层工作线程和会话状态是否已准备就绪
        if not hasattr(self, 'worker') or not self.worker.engine:
            return
            
        # 正确的属性名是 current_session_id
        current_session_id = getattr(self.worker.engine, 'current_session_id', None)
        if not current_session_id:
            return

        # 统一使用底层引擎的 NTP 同步时钟（物理基准时间），避免时间窗偏移
        if hasattr(self.worker.engine, 'time_sync'):
            current_time = self.worker.engine.time_sync.get_precise_timestamp()
        else:
            import time
            current_time = time.time()

        # 触发对齐作业
        self.align_engine.align_step(current_session_id, current_time)
