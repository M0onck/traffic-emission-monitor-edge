# ui/components/video_canvas.py
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class VideoCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.show_message("等待系统初始化...") # 初始状态

    def show_message(self, text, is_error=False):
        """显示文字提示（加载、错误、断开等）"""
        self.clear() # 清除之前的 Pixmap
        style = "color: #ff4d4d; font-weight: bold;" if is_error else "color: #aaaaaa;"
        self.setStyleSheet(f"background-color: #000000; border: 1px solid #333333; {style}")
        self.setText(text)

    def update_image(self, rgb_frame):
        """显示监控画面"""
        if rgb_frame is None: return
        
        # 只要开始更新图像，就移除背景边框和文字
        self.setStyleSheet("background-color: #000000;") 
        
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)
