import cv2
import numpy as np
import supervision as sv
from dataclasses import dataclass

"""
[表现层] 视频渲染器
功能：负责将检测结果、轨迹、车辆信息及统计数据绘制到视频帧上。
职责：
1. LabelFormatter: 负责将业务数据 (Data Objects) 格式化为人类可读的字符串。
2. Visualizer: 负责调用 OpenCV/Supervision 进行实际的图形绘制（框、标签、轨迹）。
依赖：仅依赖数据对象，不包含业务计算逻辑。
"""

@dataclass
class LabelData:
    """传输给显示层的数据对象"""
    track_id: int
    class_id: int
    speed: float = None
    display_type: str = None
    plate_points: np.ndarray = None 
    plate_color: str = None

class Visualizer:
    """
    核心渲染器
    """
    def __init__(self, calibration_points: np.ndarray, target_fps: int = 30):
        """
        初始化渲染器
        :param calibration_points: 标定区域点集
        :param trace_length: 轨迹显示长度
        :param opmode_calculator: [新增] 注入工况计算器，用于获取统一的工况描述
        """
        self.calibration_points = calibration_points.astype(np.int32)
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2, trace_length=int(target_fps * 1.0), # 保持尾迹长度为物理 1 秒的时间跨度
            position=sv.Position.BOTTOM_CENTER
        )

    def render(self, frame: np.ndarray, detections: sv.Detections, label_data_list: list, fps: float) -> np.ndarray:
        scene = frame
        
        # 1. 绘制基础图层 (ROI 区域)
        if self.calibration_points is not None and len(self.calibration_points) > 0:
            cv2.polylines(scene, [self.calibration_points], True, (0, 255, 0), 3)
            text_anchor = tuple(self.calibration_points[3]) if len(self.calibration_points) > 3 else (50, 50)
            cv2.putText(scene, "Analysis Zone", text_anchor, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. 绘制轨迹 (依然使用 supervision)
        if self.trace_annotator:
            scene = self.trace_annotator.annotate(scene=scene, detections=detections)

        # 3. 自定义绘制车辆框、标签及车牌框
        plate_color_map = {
            'blue': (255, 0, 0),      # 蓝牌
            'green': (0, 255, 0),     # 绿牌
            'yellow': (0, 255, 255),  # 黄牌
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            tid = detections.tracker_id[i] if detections.tracker_id is not None else -1
            
            # 找到对应车辆的数据对象
            data = next((d for d in label_data_list if d.track_id == tid), None)
            
            # --- 解析类型与双分类二元颜色 ---
            v_type = data.display_type if data and data.display_type else "LDV"
            
            # --- 分配视觉样式 ---
            # 只要带有 HDV (重型车) 标志，就用橙色高亮显示；否则用亮青色表示普通轻型车
            if "HDV" in v_type:
                box_color = (0, 128, 255)    # 橙色
            else:
                box_color = (255, 255, 0)    # 亮青色
            
            # 将 "LDV-Gasoline" 按照 "-" 切分，只取第一部分的 "LDV"
            short_type = v_type.split('-')[0]
            
            # 绘制车辆检测框
            cv2.rectangle(scene, (x1, y1), (x2, y2), box_color, 2)
            
            # 绘制极简标签 (例如: #5 LDV)
            label_text = f"#{tid} {short_type}"

            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(scene, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), box_color, -1)
            cv2.putText(scene, label_text, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # 绘制车牌关键点检测框
            if data and data.plate_points is not None:
                p_color = (255, 255, 255) 
                if data.plate_color:
                    p_color_str = data.plate_color.lower()
                    for k, v in plate_color_map.items():
                        if k in p_color_str:
                            p_color = v
                            break
                pts = data.plate_points.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(scene, [pts], isClosed=True, color=p_color, thickness=2)

        # 4. 在画面左上角绘制 FPS
        if fps is not None and fps > 0:
            cv2.putText(scene, f"FPS: {fps:.1f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (132, 201, 139), 3, cv2.LINE_AA)
        
        return scene

def resize_with_pad(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    工具函数：保持纵横比缩放并填充黑边
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_img
    return canvas
