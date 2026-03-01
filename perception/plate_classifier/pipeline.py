import numpy as np
from .core.typedef import *

class EdgePlateClassifier:
    """
    [边缘端适配] 轻量级车牌分类器
    功能：仅负责对传入的、已经完成透视变换和裁剪的车牌小图进行颜色分类。
    说明：目标检测和关键点提取已交由底层 Hailo 硬件完成。
    """
    def __init__(self, classifier):
        # 这里的 classifier 是 core.classification.ClassificationORT 的实例
        # 检测器 (detector) 已被彻底移除
        self.classifier = classifier

    def predict(self, pad_image: np.ndarray) -> tuple[int, float]:
        """
        执行轻量级分类推断
        Args:
            pad_image: 经过透视变换后的车牌图像 (通常为 96x96 的 np.ndarray)
        Returns:
            (plate_type, confidence): 车牌类型的枚举常数, 置信度得分
        """
        if len(pad_image.shape) != 3 or pad_image is None:
            return UNKNOWN, 0.0
            
        # 1. 直接执行分类推断 (调用 ONNXRuntime)
        cls_result = self.classifier(pad_image)
        
        # 2. 优化后的分类映射逻辑（降序寻找合法类型 + 兜底）
        plate_type = UNKNOWN
        flatten_result = cls_result.flatten()
        sorted_indices = np.argsort(flatten_result)[::-1]
        
        # 提取最高得分作为置信度
        confidence = float(flatten_result[sorted_indices[0]])
        
        for idx in sorted_indices:
            idx = int(idx)
            if idx == PLATE_TYPE_YELLOW:
                # 边缘端简化：由于去除了原先检测器的多类别输出，这里统一映射为单层黄牌。
                # 如果业务极其依赖单双层区分，后续可以根据传入图片的宽高比 (pad_image.shape) 进行辅助判断。
                plate_type = YELLOW_SINGLE 
                break
            elif idx == PLATE_TYPE_BLUE:
                plate_type = BLUE
                break
            elif idx == PLATE_TYPE_GREEN:
                plate_type = GREEN
                break
                
        # 针对黄绿牌/新能源的绝对兜底逻辑
        if plate_type == UNKNOWN:
            plate_type = GREEN

        return plate_type, confidence
