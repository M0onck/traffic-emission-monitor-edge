from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QPainterPath

# --- UI 组件：双轨物理曲线 (速度 & 加速度) ---
class SpeedCurveWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.speeds = []
        self.accels = []
        self.setMinimumHeight(200) # 稍微加高一点以容纳负半轴
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555;")

    def update_curve(self, speeds, accels):
        self.speeds = speeds
        self.accels = accels
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 1. 绘制背景
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        
        if len(self.speeds) < 2 or len(self.accels) < 2:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待车辆离场结算数据...")
            return

        # 2. 坐标系与量程设定
        # 统一速度与加速度的最大量程，强制至少为 20.0，且关于 X 轴对称
        max_val = max(20.0, max(self.speeds) * 1.2, max(abs(a) for a in self.accels) * 1.2)
        
        # Y 轴映射函数：将物理值 (-max_val 到 +max_val) 映射到像素 (h 到 0)
        def map_y(val):
            # val / max_val 的范围是 [-1, 1]
            # 屏幕坐标中，0 对应中间，+1 对应顶部(0)，-1 对应底部(h)
            return h / 2 - (val / max_val) * (h / 2)

        # 3. 绘制辅助网格线和零刻度线 (X轴)
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
        for i in range(1, 4):
            if i != 2: # 避开中间的 0 刻度线
                y_line = int(h * i / 4)
                painter.drawLine(0, y_line, w, y_line)
                
        # 绘制纯白色的 0 刻度线 (X轴)
        zero_y = int(h / 2)
        painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine))
        painter.drawLine(0, zero_y, w, zero_y)

        # 4. 构建并绘制曲线
        dx = w / max(1, len(self.speeds) - 1)
        path_spd, path_acc = QPainterPath(), QPainterPath()
        
        for i in range(len(self.speeds)):
            x = i * dx
            y_spd = map_y(self.speeds[i])
            y_acc = map_y(self.accels[i])
            
            if i == 0:
                path_spd.moveTo(x, y_spd)
                path_acc.moveTo(x, y_acc)
            else:
                path_spd.lineTo(x, y_spd)
                path_acc.lineTo(x, y_acc)

        # 绘制速度曲线 (青色)
        painter.setPen(QPen(QColor(0, 255, 255), 2))
        painter.drawPath(path_spd)
        
        # 绘制加速度曲线 (红色)
        painter.setPen(QPen(QColor(255, 50, 50), 2))
        painter.drawPath(path_acc)

        # 5. 绘制文字信息标注
        avg_spd = sum(self.speeds) / len(self.speeds)
        max_spd = max(self.speeds)
        
        painter.setFont(QFont("Arial", 11, QFont.Bold))
        
        # 左上角：图例与量程
        painter.setPen(QPen(QColor(0, 255, 255)))
        painter.drawText(10, 25, f"■ 速度 (m/s)")
        painter.setPen(QPen(QColor(255, 50, 50)))
        painter.drawText(10, 45, f"■ 加速度 (m/s²)")
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(10, 70, f"统一量程: ±{max_val:.1f}")
        
        # 右上角：速度统计信息
        painter.setPen(QPen(QColor(255, 255, 255)))
        stats_text = f"平均速度: {avg_spd:.1f} m/s   最大速度: {max_spd:.1f} m/s"
        # 获取文字宽度以靠右对齐
        text_w = painter.fontMetrics().width(stats_text)
        painter.drawText(w - text_w - 15, 25, stats_text)
