import numpy as np
import supervision as sv

class VisionPipeline:
    """
    [感知层] 视觉处理流水线
    负责从硬件缓存提取元数据、过滤噪点、NMS 抑制，以及多目标追踪。
    """
    def __init__(self, fps: int, label_map: dict):
        self.label_map = label_map
        
        # 优化 1: 调整 ByteTrack 工业级参数
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.45,  # 提高建轨门槛：只有高置信度(>0.45)的检测框才会生成新轨迹，防假阳性
            lost_track_buffer=int(fps * 3.0), # 延长记忆时间：容忍长达 3 秒的重度遮挡，防止 ID 频繁跳变
            minimum_matching_threshold=0.8,   # 保持不变，匹配 IoU 阈值
            frame_rate=fps
        )

    def process(self, frame: np.ndarray, hailo_data: list) -> sv.Detections:
        """
        处理脱壳后的纯 Python 硬件推理元数据，返回带有 Track ID 的标准 Detections 对象。
        """
        h, w = frame.shape[:2]
        xyxy, class_ids, confs = [], [], []

        for item in hailo_data:
            # 获取归一化坐标并映射到实际物理像素
            raw_x1, raw_y1 = int(item['xmin'] * w), int(item['ymin'] * h)
            raw_x2, raw_y2 = int(item['xmax'] * w), int(item['ymax'] * h)
            
            # 优化 2: 安全过滤，使用 np.clip 严格防止坐标溢出图像边界
            x1 = np.clip(min(raw_x1, raw_x2), 0, w - 1)
            y1 = np.clip(min(raw_y1, raw_y2), 0, h - 1)
            x2 = np.clip(max(raw_x1, raw_x2), 0, w - 1)
            y2 = np.clip(max(raw_y1, raw_y2), 0, h - 1)
            
            # 过滤面积过小的病态框 (防止 downstream OpenCV 报错)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            # 标签白名单过滤
            label = item['label']
            if label not in self.label_map:
                continue  
                
            cid = self.label_map[label]
            xyxy.append([x1, y1, x2, y2])
            class_ids.append(cid)
            confs.append(item['conf'])

        if len(xyxy) > 0:
            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                confidence=np.array(confs, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )
            
            # 分配追踪 ID
            detections = self.tracker.update_with_detections(detections)
        else:
            detections = sv.Detections.empty()
            detections.tracker_id = np.array([], dtype=int)
            detections.class_id = np.array([], dtype=int)
            detections.confidence = np.array([], dtype=np.float32)

        return detections
