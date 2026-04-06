# 文件路径: ui/components/edge_dialog.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QFont

class EdgeAnimatedDialog(QDialog):
    """基础动画弹窗，提供中心上下展开的科幻风动画效果"""
    def __init__(self, parent=None, target_height=220, is_warning=False):
        super().__init__(parent)
        
        # 针对 Wayland 适配：更改窗口类型为 Tool 工具层
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowState(Qt.WindowFullScreen)
        
        self.target_height = target_height
        self.v_center = 230 
        
        # 1. 全屏半透明暗黑遮罩
        self.bg = QFrame(self)
        self.bg.setGeometry(0, 0, 800, 480) # 内部绘制仍保持 800x480 的比例
        self.bg.setStyleSheet("background-color: rgba(0, 0, 0, 190);")
        
        # 2. 动画承载面板
        self.panel = QFrame(self)
        border_color = "#ff4d4f" if is_warning else "#555555" 
        self.panel.setStyleSheet(f"background-color: #0a0a0a; border-top: 2px solid {border_color}; border-bottom: 2px solid {border_color};")
        
        # 3. 动画引擎 (此时实例化它，后续 showEvent 调用就安全了)
        self.anim = QPropertyAnimation(self.panel, b"geometry")
        self.anim.setDuration(250)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        
        self.panel.setGeometry(0, self.v_center, 800, 0)
        
    def showEvent(self, event):
        super().showEvent(event)
        self.anim.setStartValue(QRect(0, self.v_center, 800, 0))
        self.anim.setEndValue(QRect(0, self.v_center - self.target_height // 2, 800, self.target_height))
        self.anim.start()
        
    def close_with_anim(self, result_code):
        self.anim.setStartValue(self.panel.geometry())
        self.anim.setEndValue(QRect(0, self.v_center, 800, 0))
        self.anim.finished.connect(lambda: self.done(result_code))
        self.anim.start()

class EdgeMessageBox(EdgeAnimatedDialog):
    """即插即用的通用消息确认弹窗"""
    def __init__(self, parent, title, text, info_text="", is_warning=False):
        super().__init__(parent, target_height=240, is_warning=is_warning)
        
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(50, 30, 50, 30)
        
        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Arial", 18, QFont.Bold))
        lbl_title.setStyleSheet(f"color: {'#ff4d4f' if is_warning else '#ffffff'}; border: none;")
        
        lbl_text = QLabel(text)
        lbl_text.setFont(QFont("Arial", 14))
        lbl_text.setStyleSheet("color: #dddddd; border: none;")
        
        layout.addWidget(lbl_title)
        layout.addSpacing(10)
        layout.addWidget(lbl_text)
        
        if info_text:
            lbl_info = QLabel(info_text)
            lbl_info.setFont(QFont("Arial", 12))
            lbl_info.setStyleSheet("color: #aaaaaa; border: none;")
            layout.addWidget(lbl_info)
            
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_cancel = QPushButton("取消")
        btn_cancel.setFixedSize(120, 45)
        btn_cancel.setFont(QFont("Arial", 14, QFont.Bold))
        btn_cancel.setStyleSheet("background-color: transparent; border: 2px solid #777; color: #fff; border-radius: 5px;")
        btn_cancel.clicked.connect(lambda: self.close_with_anim(QDialog.Rejected))
        
        btn_confirm = QPushButton("确认")
        btn_confirm.setFixedSize(120, 45)
        btn_confirm.setFont(QFont("Arial", 14, QFont.Bold))
        btn_color = "#ff4d4f" if is_warning else "#00e676"
        btn_confirm.setStyleSheet(f"background-color: transparent; border: 2px solid {btn_color}; color: {btn_color}; border-radius: 5px;")
        btn_confirm.clicked.connect(lambda: self.close_with_anim(QDialog.Accepted))
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(btn_confirm)
        layout.addLayout(btn_layout)
