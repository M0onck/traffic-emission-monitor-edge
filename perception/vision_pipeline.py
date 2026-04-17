import numpy as np
import supervision as sv

# 引入 Hailo 元数据解析库
try:
    import hailo
except ImportError:
    pass

class VisionPipeline:
    """
    [感知层] 视觉处理流水线
    负责从硬件缓存提取元数据、过滤噪点、NMS 抑制，以及多目标追踪。
    """
    def __init__(self, fps: int, label_map: dict):
        self.label_map = label_map
        
        # 初始化 Python 层的高性能 ByteTrack 追踪器
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25, 
            lost_track_buffer=int(fps * 1.0), 
            minimum_matching_threshold=0.8,
            frame_rate=fps
        )

    def process(self, frame: np.ndarray, hailo_data: list) -> sv.Detections:
        """
        处理脱壳后的纯 Python 硬件推理元数据，返回带有 Track ID 的标准 Detections 对象。
        """
        h, w = frame.shape[:2]
        xyxy, class_ids, confs = [], [], []

        # 直接遍历上一层已经解析好的纯 Python 字典列表，无需再处理底层的 buffer 和 try-except
        for item in hailo_data:
            # 获取归一化坐标并映射到实际物理像素
            raw_x1, raw_y1 = int(item['xmin'] * w), int(item['ymin'] * h)
            raw_x2, raw_y2 = int(item['xmax'] * w), int(item['ymax'] * h)
            
            # 安全过滤：确保坐标合理，防止下游崩溃
            x1, x2 = min(raw_x1, raw_x2), max(raw_x1, raw_x2)
            y1, y2 = min(raw_y1, raw_y2), max(raw_y1, raw_y2)
            
            # 标签白名单过滤
            label = item['label']
            if label not in self.label_map:
                continue  
                
            cid = self.label_map[label]
            xyxy.append([x1, y1, x2, y2])
            class_ids.append(cid)
            confs.append(item['conf'])

        # 构建并更新追踪器 (此部分逻辑保持不变)
        if len(xyxy) > 0:
            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                confidence=np.array(confs, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )
            
            # NMS 强力合并重叠框
            detections = detections.with_nms(threshold=0.4, class_agnostic=True)
            
            # 分配追踪 ID
            detections = self.tracker.update_with_detections(detections)
        else:
            detections = sv.Detections.empty()
            detections.tracker_id = np.array([], dtype=int)
            detections.class_id = np.array([], dtype=int)
            detections.confidence = np.array([], dtype=np.float32)

        return detections