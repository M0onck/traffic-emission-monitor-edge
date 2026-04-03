# 文件路径: domain/vehicle/classifier.py
import math
from collections import defaultdict

class VehicleClassifier:
    """
    [业务层] 车辆类型分类器 (重构版 - 双分体系)
    职责：基于 YOLO 物理过滤后的类别和 OCR 车牌颜色，输出标准的微观排放大类
    (如 'LDV-Gasoline', 'LDV-Electric', 'HDV-Diesel', 'HDV-Electric')。
    """
    def __init__(self, type_map=None, yolo_classes=None):
        """
        :param type_map: 废弃参数，保留仅为了兼容旧的外部调用代码，防止报错
        :param yolo_classes: 类别ID配置，默认为 COCO 标准 {'car': 2, 'bus': 5, 'truck': 7}
        """
        yolo_classes = yolo_classes or {'car': 2, 'bus': 5, 'truck': 7}
        self.cls_car = yolo_classes.get('car', 2)
        self.cls_bus = yolo_classes.get('bus', 5)
        self.cls_truck = yolo_classes.get('truck', 7)

    def resolve_type(self, class_id, plate_history=None, plate_color_override=None):
        """
        统一的车辆类型与能源判定逻辑
        :return: (final_color, final_type_string)
        """
        # 1. 确定车牌颜色 (颜色是推断能源类型最可靠的物理特征)
        final_color = "Unknown"
        if plate_color_override and plate_color_override != "Unknown":
            final_color = plate_color_override
        elif plate_history:
            # 加权投票逻辑： Conf * sqrt(Area)
            scores = defaultdict(float)
            for e in plate_history:
                conf = e.get('conf', 1.0)
                area = e.get('area', 0.0)
                weight = conf * math.sqrt(area)
                scores[e['color']] += weight
                
            if scores: 
                final_color = max(scores, key=scores.get)
            
        # 2. 核心双分逻辑：结合物理过滤后的 YOLO ID 与车牌颜色，输出终极类型
        
        # --- LDV 轻型车逻辑 (普通轿车、SUV、面包车、皮卡) ---
        if class_id == self.cls_car:
            if final_color == "green":
                return final_color, "LDV-Electric" # 绿牌小车 -> 轻型新能源
            else:
                return final_color, "LDV-Gasoline" # 蓝牌/未知 -> 默认轻型燃油车

        # --- HDV 重型车逻辑 (重卡、大客车) ---
        elif class_id in [self.cls_bus, self.cls_truck]:
            if final_color == "green":
                return final_color, "HDV-Electric" # 绿牌重型车 -> 重型电动/混动
            else:
                return final_color, "HDV-Diesel"   # 黄牌/未知 -> 默认重型柴油车
                
        # 3. 兜底保护 (防止异常类别 ID 导致系统崩溃)
        return final_color, "LDV-Gasoline"
