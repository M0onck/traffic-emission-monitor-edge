# 文件路径: domain/vehicle/physical_filter.py
import numpy as np

class PhysicalVehicleFilter:
    """
    [领域层] 物理与几何属性过滤器
    职责：结合 2D 像素特征与 IPM 逆透视物理尺寸，对目标进行前置清洗和类别修正。
    """
    def __init__(self, cfg):
        # --- 像素级过滤阈值 ---
        self.max_area_ratio = 0.6      # 最大面积比例 (防全屏幽灵框)
        self.min_aspect_ratio = 0.3    # 最小长宽比 (防狭长地标/箭头)
        self.max_aspect_ratio = 3.5    # 最大长宽比 (防横向异常色块)
        
        # --- 物理级过滤阈值 ---
        # 设定重型车辆(HDV)的最小物理截面面积阈值 (平方米)
        self.hdv_min_surface_area_m2 = 10.0
        
        # 兼容当前的 COCO 类别 ID
        # 假设配置中未定义则使用默认的 COCO ID: 2(car), 5(bus), 7(truck)
        self.cls_car = getattr(cfg, 'YOLO_CLASS_CAR', 2)
        self.cls_bus = getattr(cfg, 'YOLO_CLASS_BUS', 5)
        self.cls_truck = getattr(cfg, 'YOLO_CLASS_TRUCK', 7)

    def apply_pixel_filters(self, detections, frame_shape):
        """滤除不符合车辆基本像素几何规律的噪点 (幽灵框、箭头)"""
        if len(detections) == 0:
            return detections
            
        h, w = frame_shape[:2]
        
        box_ws = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        box_hs = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        box_areas = box_ws * box_hs
        
        box_hs = np.maximum(box_hs, 1e-5)
        aspect_ratios = box_ws / box_hs
        
        mask_area = box_areas < (w * h * self.max_area_ratio)
        mask_ratio = (aspect_ratios > self.min_aspect_ratio) & (aspect_ratios < self.max_aspect_ratio)
        
        valid_mask = mask_area & mask_ratio
        return detections[valid_mask]

    def correct_classes_by_physics(self, detections, spatial_analyzer):
        """利用深度归一化的伪物理表面积，将误判的 SUV/面包车降级为轻型车"""
        if len(detections) == 0:
            return detections
            
        for i, (cls_id, xyxy) in enumerate(zip(detections.class_id, detections.xyxy)):
            if cls_id in [self.cls_truck, self.cls_bus]:
                x1, y1, x2, y2 = xyxy
                
                # 1. 纯像素面积
                pixel_area = (x2 - x1) * (y2 - y1)
                
                # 2. 提取底边中心点及向右偏移 1 像素的点 (Z=0)
                bc_x = (x1 + x2) / 2.0
                bc_pt = np.array([bc_x, y2])
                bc_pt_right = np.array([bc_x + 1.0, y2])
                
                # 3. 物理投射
                bc_phys = spatial_analyzer.get_physical_point(bc_pt)
                bc_right_phys = spatial_analyzer.get_physical_point(bc_pt_right)
                
                if bc_phys is not None and bc_right_phys is not None:
                    # 4. 计算当前距离下的 1 像素物理当量 (米/像素)
                    meters_per_pixel = np.sqrt((bc_phys[0] - bc_right_phys[0])**2 + (bc_phys[1] - bc_right_phys[1])**2)
                    
                    # 5. 伪物理表面积 (平方米)
                    pseudo_surface_area_m2 = pixel_area * (meters_per_pixel ** 2)
                    
                    # 6. 终极判决：面积不达标，强制剥夺重型车身份
                    if pseudo_surface_area_m2 < self.hdv_min_surface_area_m2:
                        detections.class_id[i] = self.cls_car 
                        
        return detections
