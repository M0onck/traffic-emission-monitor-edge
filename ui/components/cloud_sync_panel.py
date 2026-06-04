#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QStackedLayout, 
                             QLabel, QComboBox, QPushButton, QFrame, QGridLayout, 
                             QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

class LoadingWidget(QWidget):
    """
    网络检测/加载动画界面（仅限文本模拟转圈）
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # 旋转动画状态字符
        self.frames = ['|', '/', '-', '\\']
        self.frame_idx = 0
        
        # 动画标签
        self.spinner_label = QLabel(self.frames[0])
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_label.setStyleSheet("font-size: 56px; font-weight: bold; color: #4CAF50;")
        
        # 提示文本
        self.status_label = QLabel("正在检查网络连通状态...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: #DDDDDD; margin-top: 20px;")
        
        self.layout.addStretch()
        self.layout.addWidget(self.spinner_label)
        self.layout.addWidget(self.status_label)
        self.layout.addStretch()
        
        # 动画定时器 (每 150ms 切换一帧)
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._update_spinner)
        
    def start_animation(self):
        self.anim_timer.start(150)
        
    def stop_animation(self):
        self.anim_timer.stop()
        
    def _update_spinner(self):
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)
        self.spinner_label.setText(self.frames[self.frame_idx])


class CloudSyncDashboard(QWidget):
    """
    云端同步设置与数据概览主界面
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        # 【优化 1】整体边距稍微缩小，给中间表格留出更多高度
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(15)
        
        # 1. 顶部控制栏 (设置项)
        control_frame = QFrame()
        control_frame.setStyleSheet("QFrame { background-color: #2a2a2a; border-radius: 8px; }")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(15, 8, 15, 8) # 上下边距缩紧
        
        title_label = QLabel("云端同步")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")

        # 状态指示 Badge
        self.status_badge = QLabel("未开启")
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedHeight(24)
        self.status_badge.setMinimumWidth(60)
        self.status_badge.setStyleSheet("""
            QLabel { 
                background-color: #666666; 
                color: white; 
                border-radius: 4px; 
                font-size: 12px; 
                font-weight: bold; 
            }
        """)
        
        # 下拉菜单 - 上传间隔
        interval_label = QLabel("上传间隔:")
        interval_label.setStyleSheet("font-size: 13px; color: #BBBBBB;")
        
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["10 秒", "30 秒", "60 秒"])
        self.interval_combo.setStyleSheet("""
            QComboBox { background: #3c3c3c; color: white; border: 1px solid #555; padding: 4px; border-radius: 4px; }
            QComboBox::drop-down { border: 0px; }
        """)
        
        # 开关按钮 (Toggle)
        self.sync_toggle_btn = QPushButton("开启同步")
        self.sync_toggle_btn.setCheckable(True)
        self.sync_toggle_btn.setFixedSize(110, 32)
        self.sync_toggle_btn.toggled.connect(self._on_toggle_sync)
        self._on_toggle_sync(False) # 初始化样式
        
        control_layout.addWidget(title_label)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.status_badge)
        control_layout.addStretch()
        control_layout.addWidget(interval_label)
        control_layout.addWidget(self.interval_combo)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.sync_toggle_btn)
        
        self.layout.addWidget(control_frame)
        
        # 2. 中部数据概览区 (3x3 表格)
        data_title = QLabel("最近一次上传的对齐快照 (Aligned Snapshot)")
        data_title.setStyleSheet("font-size: 13px; color: #AAAAAA; margin-top: 5px;")
        self.layout.addWidget(data_title)
        
        grid_frame = QFrame()
        self.grid_layout = QGridLayout(grid_frame)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(10) # 【优化 2】缩小卡片间隙
        
        self.value_labels = {}
        
        # --- 交通属性 (Row 0) ---
        self._add_card(0, 0, "车辆总数", "veh_count", "0", "辆")
        self._add_card(0, 1, "小型车 (LDV)", "ldv_count", "0", "辆")
        self._add_card(0, 2, "大型车 (HDV)", "hdv_count", "0", "辆")
        
        # --- 气象与环境 1 (Row 1) ---
        self._add_card(1, 0, "温度", "temp", "--", "°C")
        self._add_card(1, 1, "湿度", "humidity", "--", "%")
        self._add_card(1, 2, "PM2.5", "pm25", "--", "μg/m³")
        
        # --- 气象与环境 2 (Row 2) ---
        self._add_card(2, 0, "风速", "wind_speed", "--", "m/s")
        self._add_card(2, 1, "风向", "wind_dir", "--", "°")
        self._add_card(2, 2, "PM10", "pm10", "--", "μg/m³")
        
        # 【优化 3】赋予 grid_frame 弹性系数 stretch=1，使其填满剩余屏幕，
        # 并删除了底部多余的 self.layout.addStretch() 避免纵向挤压
        self.layout.addWidget(grid_frame, stretch=1)

    def _add_card(self, row, col, title, key, default_val, unit):
        """生成 3x3 表格中的独立圆角卡片"""
        card = QFrame()
        # 允许卡片在网格中自由纵横向拉伸
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.setStyleSheet("QFrame { background-color: #333333; border: 1px solid #444; border-radius: 8px; }")
        
        card_layout = QVBoxLayout(card)
        # 【优化 4】将卡片内部 Padding 从 15 缩减到 8，给文字留出渲染空间
        card_layout.setContentsMargins(10, 8, 10, 8) 
        card_layout.setSpacing(2)
        
        t_label = QLabel(title)
        t_label.setAlignment(Qt.AlignCenter)
        t_label.setStyleSheet("color: #999999; font-size: 12px; border: none;") # 缩减 1px
        
        v_layout = QHBoxLayout()
        v_layout.setContentsMargins(0, 0, 0, 0)
        
        v_label = QLabel(default_val)
        v_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # 【优化 5】核心数据字号从 24px 降为 20px，防止数字过大被截断
        v_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold; border: none;") 
        
        u_label = QLabel(unit)
        u_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        u_label.setStyleSheet("color: #777777; font-size: 11px; border: none; padding-bottom: 3px;")
        
        v_layout.addWidget(v_label)
        v_layout.addWidget(u_label)
        v_layout.setStretch(0, 1)
        
        card_layout.addWidget(t_label)
        card_layout.addLayout(v_layout)
        
        self.grid_layout.addWidget(card, row, col)
        
        self.value_labels[key] = v_label

    def _on_toggle_sync(self, checked):
        """处理开关按钮的 UI 样式"""
        if checked:
            self.sync_toggle_btn.setText("关闭同步")
            self.sync_toggle_btn.setStyleSheet("""
                QPushButton { background-color: #f44336; color: white; border-radius: 4px; font-weight: bold;}
                QPushButton:hover { background-color: #d32f2f; }
            """)
        else:
            self.sync_toggle_btn.setText("开启同步")
            self.sync_toggle_btn.setStyleSheet("""
                QPushButton { background-color: #4CAF50; color: white; border-radius: 4px; font-weight: bold;}
                QPushButton:hover { background-color: #388E3C; }
            """)

    def update_status(self, status_str):
        """供 Controller 调用，更新运行状态标签颜色"""
        self.status_badge.setText(status_str)
        if status_str == "运行中":
            self.status_badge.setStyleSheet("QLabel { background-color: #4CAF50; color: white; border-radius: 4px; font-size: 12px; font-weight: bold; }")
        elif status_str == "待命中":
            self.status_badge.setStyleSheet("QLabel { background-color: #f39c12; color: white; border-radius: 4px; font-size: 12px; font-weight: bold; }")
        else:
            self.status_badge.setStyleSheet("QLabel { background-color: #666666; color: white; border-radius: 4px; font-size: 12px; font-weight: bold; }")

    def update_data(self, data_dict):
        """供外部业务逻辑调用，刷新 UI 数据"""
        for key, val in data_dict.items():
            if key in self.value_labels:
                self.value_labels[key].setText(str(val))


class CloudSyncPanel(QWidget):
    """
    云端同步功能的入口组件 (包含连通性检测流)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stacked_layout = QStackedLayout(self)
        
        self.loading_page = LoadingWidget()
        self.dashboard_page = CloudSyncDashboard()
        
        self.stacked_layout.addWidget(self.loading_page)
        self.stacked_layout.addWidget(self.dashboard_page)
        
    def showEvent(self, event):
        """当面板显示时触发检测流程"""
        super().showEvent(event)
        self._start_connection_check()
        
    def _start_connection_check(self):
        """执行界面连通性模拟（无业务逻辑版）"""
        self.stacked_layout.setCurrentWidget(self.loading_page)
        self.loading_page.start_animation()
        
        # 模拟2秒的网络延迟后，显示正常Dashboard界面
        QTimer.singleShot(2000, self._on_connection_success)
        
    def _on_connection_success(self):
        self.loading_page.stop_animation()
        self.stacked_layout.setCurrentWidget(self.dashboard_page)
