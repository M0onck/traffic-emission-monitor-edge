import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

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
