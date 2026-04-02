import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QImage, QPixmap

from ui.workers.engine_worker import EngineWorker

class MainController:
    """Controller 层：负责状态管理、页面路由、信号绑定与定时更新"""
    def __init__(self, view):
        self.view = view
        self.is_collecting = False
        self.sampled_tid = None
        self.worker = None
        
        self.dash_timer = QTimer(self.view)
        self.dash_timer.timeout.connect(self.update_timer_tasks)

        self.bind_signals()
        self.view.close_callback = self.cleanup  # 注入关闭事件钩子
        self.update_nav_buttons()

    def bind_signals(self):
        """将视图组件的事件绑定到控制器的逻辑上"""
        self.view.btn_home.clicked.connect(self.return_to_home)
        self.view.btn_stop.clicked.connect(self.stop_collection_trigger)
        self.view.btn_prev.clicked.connect(self.prev_page)
        self.view.btn_next.clicked.connect(self.next_page)
        self.view.btn_app1.clicked.connect(self.route_app1_click)
        self.view.btn_exit.clicked.connect(self.view.close)
    
    def route_app1_click(self):
        """主界面按钮的智能跳转路由"""
        if self.is_collecting:
            # 如果已经在采集中，直接跳过标定和设置，切入监控面板 (Index 3)
            self.enter_app(3)
        else:
            # 如果尚未运行，按照正常流程进入第一步标定环节 (Index 1)
            self.enter_app(1)
    
    def enter_app(self, target_idx):
        """进入具体功能的槽函数"""
        self.stack.setCurrentIndex(target_idx)
        self.update_nav_buttons()

    def return_to_home(self):
        """返回主界面"""
        self.stack.setCurrentIndex(0)
        self.update_nav_buttons()
        self.update_main_menu_btn_style()

    def prev_page(self):
        """上一页触发逻辑"""
        idx = self.stack.currentIndex()
        if idx > 1: # 防止用户通过“上一步”按钮退回到主菜单
            self.stack.setCurrentIndex(idx - 1)
        self.update_nav_buttons()

    def next_page(self):
        """下一页触发逻辑"""
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
    
    # --- 任务启停控制 ---
    def start_engine(self):
        # 提取的是反算好的原生 1080p 真实坐标
        source_points = self.canvas.get_real_points()
        
        # 启动后台引擎线程
        self.worker = EngineWorker(source_points, self.phys_w, self.phys_h)
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.start()
    
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
    
    def cleanup(self):
        """处理整个应用的关闭回收"""
        if self.is_collecting:
            self.final_stop_process()
    
    # --- 界面渲染更新 ---
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
