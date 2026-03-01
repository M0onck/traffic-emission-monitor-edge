import numpy as np
from .core.typedef import *
from .core.tools_process import get_rotate_crop_image

class EdgePlateClassifierPipeline:
    """
    [边缘端修正版] 车牌识别全流程
    在 Python 端接收车辆小图，使用 y5fu 找牌，透视变换后用 litemodel 分类。
    """
    def __init__(self, detector, classifier):
        # 这里的 detector 就是原版的 MultiTaskDetectorORT (跑 y5fu)
        # 这里的 classifier 就是原版的 ClassificationORT (跑 litemodel)
        self.detector = detector
        self.classifier = classifier

    def process(self, vehicle_crop: np.ndarray):
        """
        处理单张车辆裁剪图
        """
        # 1. 寻找车牌及关键点
        bboxes, landmarks = self.detector(vehicle_crop)
        if len(bboxes) == 0:
            return UNKNOWN, 0.0, None
            
        # 取置信度最高的车牌框
        best_idx = np.argmax(bboxes[:, 4])
        plate_box = bboxes[best_idx]
        plate_points = landmarks[best_idx]
        
        # 2. 透视变换抠图
        pad_image = get_rotate_crop_image(vehicle_crop, plate_points)
        
        # 3. 车牌分类
        cls_result = self.classifier(pad_image)
        plate_type = UNKNOWN
        flatten_result = cls_result.flatten()
        sorted_indices = np.argsort(flatten_result)[::-1]
        confidence = float(flatten_result[sorted_indices[0]])
        
        # 简化的类型映射...
        idx = int(sorted_indices[0])
        if idx == PLATE_TYPE_YELLOW: plate_type = YELLOW_SINGLE 
        elif idx == PLATE_TYPE_BLUE: plate_type = BLUE
        elif idx == PLATE_TYPE_GREEN: plate_type = GREEN
        if plate_type == UNKNOWN: plate_type = GREEN

        return plate_type, confidence, plate_box
