import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class CalibrationCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.orig_frame = None
        self.real_points = []
        self.drag_idx = -1
        self.selected_idx = -1  
        self.dpad_rects = {}    
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 380)
        
        self.shared_camera = None
        
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._read_next_frame)

    def load_camera(self, camera_instance):
        """接收外部注入的全局 Camera 实例"""
        self.stop_preview()
        time.sleep(0.1) 
        self.shared_camera = camera_instance

        # 确保管道已启动
        if not self.shared_camera.is_running:
            self.shared_camera.start()
        
        print(">>> [DEBUG] 正在通过全局 GstPipelineManager 拉取预处理画面...")
        frame_received = False
        for _ in range(100):  # 最长容忍等待 10 秒
            # 解构返回的元组 (clean_frame, buffer_meta)，标定只要干净画面
            frame, _ = self.shared_camera.read()
            if frame is not None:
                self._start_preview(frame)
                frame_received = True
                break
            time.sleep(0.1)
        
        if not frame_received:
            print(">>> 警告：视频流拉取超时！")
            self._show_error_screen("Video Stream Error")

    def _start_preview(self, first_frame):
        """成功读取第一帧后的初始化工作"""
        # 记录当前已存在的画面分辨率
        old_shape = self.orig_frame.shape[:2] if self.orig_frame is not None else None
        
        self.orig_frame = first_frame
        new_shape = self.orig_frame.shape[:2]
        
        # 触发条件：如果是首次加载，或者视频源分辨率发生了变动，则重新初始化标定框
        if not self.real_points or old_shape != new_shape: 
            self.init_points()
            # 如果发生了重置，清除可能存在的选中状态，防止放大镜热区越界
            self.selected_idx = -1
            self.drag_idx = -1
            
        self.update_display()
        self.preview_timer.start(33)

    def _show_error_screen(self, msg="摄像头离线..."):
        """绘制错误画面"""
        self.orig_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(self.orig_frame, msg, (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        self.update_display()

    def _read_next_frame(self):
        """
        定时器回调：直接向统一的 camera 接口索要画面
        """
        if self.shared_camera and self.shared_camera.is_running:
            frame, _ = self.shared_camera.read()
            if frame is not None:
                self.orig_frame = frame
                self.update_display()

    def stop_preview(self):
        """
        停止预览。
        注意：不要在这里调用 shared_camera.stop()
        生命周期已上移，这里只负责停止 UI 定时刷新，让管道在后台继续热身。
        """
        self.preview_timer.stop()

    def init_points(self):
        h, w = self.orig_frame.shape[:2]
        cx, cy = w // 2, h // 2
        dx, dy = int(w * 0.2), int(h * 0.2)
        self.real_points = [[cx - dx, cy + dy], [cx + dx, cy + dy], 
                            [cx + dx, cy - dy], [cx - dx, cy - dy]]

    def _get_transform_params(self):
        cw, ch = self.width(), self.height()
        if cw < 100 or ch < 100:
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
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        resized = cv2.resize(self.orig_frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
        
        pts_ui = []
        for rx, ry in self.real_points:
            ux = int(rx * scale + x_off)
            uy = int(ry * scale + y_off)
            pts_ui.append([ux, uy])
            
        pts_ui_arr = np.array(pts_ui, np.int32)
        cv2.polylines(canvas, [pts_ui_arr], True, (0, 255, 0), 2)
        
        labels = ["BL", "BR", "TR", "TL"]
        for i, (ux, uy) in enumerate(pts_ui):
            # 选中点标记为醒目的橙色，普通点为绿色
            color = (0, 165, 255) if i == self.selected_idx else (0, 255, 0)
            cv2.circle(canvas, (ux, uy), 15, color, -1)
            # 被选中的点增加一层空心外圈，加强视觉反馈
            if i == self.selected_idx:
                cv2.circle(canvas, (ux, uy), 22, (0, 255, 255), 2)
            cv2.putText(canvas, labels[i], (ux+20, uy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ====== 如果选中了点，绘制局部放大镜和中心方向键 ======
        self.dpad_rects.clear()
        if self.selected_idx != -1:
            # 1. 绘制局部放大镜 (Loupe)
            rx, ry = int(self.real_points[self.selected_idx][0]), int(self.real_points[self.selected_idx][1])
            roi_size = 40  # 从原图中截取的区域大小
            zoom_fac = 4   # 放大倍数
            oh, ow = self.orig_frame.shape[:2]
            
            y1, y2 = max(0, ry - roi_size//2), min(oh, ry + roi_size//2)
            x1, x2 = max(0, rx - roi_size//2), min(ow, rx + roi_size//2)
            roi = self.orig_frame[y1:y2, x1:x2].copy()
            
            if roi.size > 0:
                # 使用 INTER_NEAREST 保持像素颗粒感，方便精确对齐边缘
                zoomed = cv2.resize(roi, (0,0), fx=zoom_fac, fy=zoom_fac, interpolation=cv2.INTER_NEAREST)
                zh, zw = zoomed.shape[:2]
                
                # 画放大镜中心的十字准星
                cv2.line(zoomed, (zw//2, 0), (zw//2, zh), (0, 255, 255), 1)
                cv2.line(zoomed, (0, zh//2), (zw, zh//2), (0, 255, 255), 1)
                
                # 智能位置：如果用户点在右边，放大镜显示在左边，防止手指遮挡
                ux = pts_ui[self.selected_idx][0]
                mx = 20 if ux > cw // 2 else cw - zw - 20
                my = 20
                
                canvas[my:my+zh, mx:mx+zw] = zoomed
                cv2.rectangle(canvas, (mx, my), (mx+zw, my+zh), (0, 165, 255), 2)
                cv2.putText(canvas, "Pixel View", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # 2. 绘制屏幕中心方向键 (D-Pad)
            cx, cy = cw // 2, ch // 2
            btn_s = 60 # 按钮尺寸 (适合触摸屏)
            gap = 10
            positions = {
                'U': (cx, cy - btn_s - gap),
                'D': (cx, cy + btn_s + gap),
                'L': (cx - btn_s - gap, cy),
                'R': (cx + btn_s + gap, cy)
            }
            
            for key, (bx, by) in positions.items():
                bx1, by1 = bx - btn_s//2, by - btn_s//2
                bx2, by2 = bx + btn_s//2, by + btn_s//2
                self.dpad_rects[key] = (bx1, by1, bx2, by2) # 保存热区供点击检测
                
                # 画按钮底色和边框
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (200, 200, 200), 2)
                
                # 画箭头符号
                text = {'U': '^', 'D': 'v', 'L': '<', 'R': '>'}[key]
                cv2.putText(canvas, text, (bx1 + 20, by1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        # ==============================================================

        rgb_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch_ = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, ch_ * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, event):
        if self.orig_frame is None: return
        x, y = event.x(), event.y()
        
        # 1. 优先检测是否点击了方向键 (微调功能)
        if self.selected_idx != -1:
            for key, (x1, y1, x2, y2) in self.dpad_rects.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    rx, ry = self.real_points[self.selected_idx]
                    # 针对原分辨率执行像素级微调
                    if key == 'U': ry -= 1
                    elif key == 'D': ry += 1
                    elif key == 'L': rx -= 1
                    elif key == 'R': rx += 1
                    
                    oh, ow = self.orig_frame.shape[:2]
                    self.real_points[self.selected_idx] = [max(0, min(ow, rx)), max(0, min(oh, ry))]
                    self.update_display()
                    return # 拦截事件，不再继续执行选中/拖拽检测

        # 2. 检测是否选中或拖拽角点
        scale, x_off, y_off, _, _, _, _ = self._get_transform_params()
        hit_radius = 50  # 加大了热区，更容易触控命中
        hit_found = False
        
        for i, (rx, ry) in enumerate(self.real_points):
            ux = rx * scale + x_off
            uy = ry * scale + y_off
            if np.hypot(x - ux, y - uy) < hit_radius:
                self.drag_idx = i
                self.selected_idx = i # 设置选中状态
                hit_found = True
                self.update_display()
                break
                
        if not hit_found:
            # 点击了空白区域，取消选中
            self.selected_idx = -1
            self.update_display()

    def mouseMoveEvent(self, event):
        if self.drag_idx != -1 and self.orig_frame is not None:
            x, y = event.x(), event.y()
            scale, x_off, y_off, _, _, cw, ch = self._get_transform_params()
            
            rx = (x - x_off) / scale
            ry = (y - y_off) / scale
            
            oh, ow = self.orig_frame.shape[:2]
            self.real_points[self.drag_idx] = [max(0, min(ow, rx)), max(0, min(oh, ry))]
            self.selected_idx = self.drag_idx # 拖动时保持选中状态
            self.update_display()

    def mouseReleaseEvent(self, event):
        self.drag_idx = -1
        # 注意：这里不清空 selected_idx，保持方向键显示
        self.update_display()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.orig_frame is not None:
            self.update_display()

    def get_real_points(self):
        pts = np.array(self.real_points, dtype=np.float32)
        oh, ow = self.orig_frame.shape[:2]
        pts[:, 0] /= ow
        pts[:, 1] /= oh
        return pts
