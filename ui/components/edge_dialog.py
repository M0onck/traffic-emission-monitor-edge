# 文件路径: ui/components/edge_dialog.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QCheckBox, QScrollArea, QWidget
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QFont

class EdgeAnimatedDialog(QDialog):
    """基础动画弹窗，提供中心上下展开的科幻风动画效果"""
    def __init__(self, parent=None, target_height=220, is_warning=False):
        super().__init__(parent)
        
        # Wayland 显示适配
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowState(Qt.WindowFullScreen)
        
        self.target_height = target_height
        self.v_center = 230 
        
        # 1. 全屏半透明暗黑遮罩
        self.bg = QFrame(self)
        self.bg.setGeometry(0, 0, 800, 480) 
        self.bg.setStyleSheet("background-color: rgba(0, 0, 0, 190);")
        
        # 2. 动画承载面板
        self.panel = QFrame(self)
        border_color = "#ff4d4f" if is_warning else "#555555" 
        self.panel.setStyleSheet(f"background-color: #0a0a0a; border-top: 2px solid {border_color}; border-bottom: 2px solid {border_color};")
        
        # 3. 动画引擎
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
    
    def mousePressEvent(self, event):
        """处理鼠标/触摸点击事件"""
        # 获取点击坐标，并判断是否在中央面板的矩形区域外
        if not self.panel.geometry().contains(event.pos()):
            # 如果在面板外（即点击了上下黑色的半透明遮罩），直接触发带有动画的取消操作
            self.close_with_anim(QDialog.Rejected)
        else:
            # 如果点在面板内部，则正常传递事件给子控件（如按钮）
            super().mousePressEvent(event)

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

class EdgeExportDialog(EdgeAnimatedDialog):
    """用于选择采集任务并导出视频的多选弹窗"""
    def __init__(self, parent, session_data_map):
        # 动态计算高度，最多不超过 350
        height = min(350, 180 + len(session_data_map) * 40)
        super().__init__(parent, target_height=height)
        
        self.selected_sessions = []
        
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(40, 20, 40, 20)
        
        title = QLabel("选择需要导出的采集任务")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffffff; border: none;")
        layout.addWidget(title)
        layout.addSpacing(10)
        
        # 使用滚动区域容纳可能很多的任务
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        content = QWidget()
        content.setStyleSheet("background-color: transparent;")
        self.vbox = QVBoxLayout(content)
        self.vbox.setSpacing(10)
        
        self.checkboxes = {}
        for session_id, count in session_data_map.items():
            # 显示任务ID和包含的视频切片数量
            cb = QCheckBox(f"任务: {session_id} (含 {count} 个视频)")
            cb.setFont(QFont("Arial", 13))
            cb.setStyleSheet("QCheckBox { color: #dddddd; } QCheckBox::indicator { width: 20px; height: 20px; }")
            self.checkboxes[session_id] = cb
            self.vbox.addWidget(cb)
            
        self.vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_cancel = QPushButton("取消")
        btn_cancel.setFixedSize(100, 40)
        btn_cancel.setStyleSheet("background-color: transparent; border: 2px solid #777; color: #fff; border-radius: 5px;")
        btn_cancel.clicked.connect(lambda: self.close_with_anim(QDialog.Rejected))
        
        btn_confirm = QPushButton("开始导出")
        btn_confirm.setFixedSize(100, 40)
        btn_confirm.setStyleSheet("background-color: transparent; border: 2px solid #00e676; color: #00e676; border-radius: 5px;")
        btn_confirm.clicked.connect(self.accept_selection)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addSpacing(15)
        btn_layout.addWidget(btn_confirm)
        layout.addLayout(btn_layout)

    def accept_selection(self):
        self.selected_sessions = [sid for sid, cb in self.checkboxes.items() if cb.isChecked()]
        self.close_with_anim(QDialog.Accepted)
