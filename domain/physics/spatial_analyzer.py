import numpy as np
import supervision as sv

class SpatialAnalyzer:
    """
    [领域服务] 空间分析器
    负责像素/物理坐标系转换、ROI 包含判定、纯几何距离计算。
    """
    def __init__(self):
        self.transformer = None

    def set_transformer(self, transformer):
        """注入底层视图转换器"""
        self.transformer = transformer

    def is_in_roi(self, point: list) -> bool:
        """判定像素点是否在分析区域内"""
        if not self.transformer:
            return False
        return self.transformer.is_in_roi(point)

    def get_physical_point(self, pixel_point: list) -> np.ndarray:
        """单点像素坐标转物理坐标"""
        if not self.transformer:
            return np.array(pixel_point)
        return self.transformer.transform_points(np.array([pixel_point]))[0]

    def get_dynamic_tolerance(self, pixel_point: list, pixel_offset: int = 2) -> float:
        """
        计算指定像素点的动态物理宽容度（用于外部死区判定）
        探针法：在 Y 轴增加微小像素偏移，计算对应的物理距离差
        """
        if not self.transformer:
            return 0.0
        curr_phys = self.get_physical_point(pixel_point)
        probe_phys = self.get_physical_point([pixel_point[0], pixel_point[1] + pixel_offset])
        return abs(curr_phys[1] - probe_phys[1])

    def calculate_geometric_distance(self, physical_points: list) -> float:
        """
        计算一组 2D 物理坐标点的纯几何折线总长度（无视时间戳）
        """
        if len(physical_points) < 2:
            return 0.0
        pts_array = np.array(physical_points)
        diffs = pts_array[1:] - pts_array[:-1]
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
