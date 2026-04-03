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

    def process(self, frame: np.ndarray, buffer) -> sv.Detections:
        """
        处理单帧硬件推理输出，返回带有 Track ID 的标准 Detections 对象。
        """
        h, w = frame.shape[:2]
        xyxy, class_ids, confs = [], [], []

        try:
            roi = hailo.get_roi_from_buffer(buffer)
            hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            
            for det in hailo_detections:
                bbox = det.get_bbox()
                raw_x1, raw_y1 = int(bbox.xmin() * w), int(bbox.ymin() * h)
                raw_x2, raw_y2 = int(bbox.xmax() * w), int(bbox.ymax() * h)
                
                # 安全过滤：确保坐标合理，防止下游崩溃
                x1, x2 = min(raw_x1, raw_x2), max(raw_x1, raw_x2)
                y1, y2 = min(raw_y1, raw_y2), max(raw_y1, raw_y2)
                
                # 标签白名单过滤
                label = det.get_label()
                if label not in self.label_map:
                    continue  
                    
                cid = self.label_map[label]
                xyxy.append([x1, y1, x2, y2])
                class_ids.append(cid)
                confs.append(det.get_confidence())

        except Exception:
            # 解析出错或在非真实设备环境下，静默通过
            pass
            
        # 构建并更新追踪器
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
