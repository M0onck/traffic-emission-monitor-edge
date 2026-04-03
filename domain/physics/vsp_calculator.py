# 文件路径: domain/physics/vsp_calculator.py

class VSPCalculator:
    """
    [业务层] 车辆比功率 (Vehicle Specific Power, VSP) 计算器 (LDV/HDV 双分类重构版)

    **职责 (Responsibility):**
    根据车辆的瞬时运动状态（速度、加速度）和物理参数，计算车辆的比功率。
    VSP 是衡量车辆发动机负载的代理变量，广泛应用于 EPA MOVES 等排放模型中，
    用于查找对应的运行工况 (OpMode) 和排放因子。

    **物理公式 (Physical Formula):**
    VSP 定义为发动机输出功率与车辆质量的比值（单位: $kW/tonne$）。
    计算公式如下：

    $$ \text{VSP} = v \cdot [1.1 \cdot a + 9.81 \cdot \sin(\theta)] + (a_m \cdot v + b_m \cdot v^2 + c_m \cdot v^3) $$
    """

    def __init__(self, config: dict):
        """
        初始化 VSP 计算器。
        """
        # 道路坡度 (默认为 0.0)
        self.road_grade = config.get("road_grade_percent", 0.0)
        
        # 允许通过外部 config 覆盖系数，否则使用内置的 EPA 标准推荐值
        custom_coeffs = config.get("vsp_coefficients", {})
        
        # LDV (轻型车) 物理阻力系数 (轿车/SUV/轻客/皮卡)
        # a_m: 滚动阻力, b_m: 旋转阻力, c_m: 空气阻力
        self.ldv_coeffs = custom_coeffs.get("LDV", 
            {"a_m": 0.156, "b_m": 0.002, "c_m": 0.0005}
        )
        
        # HDV (重型车) 物理阻力系数 (中重卡/大客车)
        # 重型车由于迎风面积巨大，空气阻力系数(c_m)显著偏高；
        # 但按总吨位归一化后的滚动阻力(a_m)通常比轻型车小。
        self.hdv_coeffs = custom_coeffs.get("HDV", 
            {"a_m": 0.089, "b_m": 0.0, "c_m": 0.0014}
        )

    def calculate(self, v_ms: float, a_ms2: float, vehicle_type: str) -> float:
        """
        计算单帧的车辆比功率 (VSP)。

        Args:
            v_ms (float): 瞬时速度，单位 $m/s$。
            a_ms2 (float): 瞬时加速度，单位 $m/s^2$。
            vehicle_type (str): 车辆类型字符串 (如 'LDV-Gasoline', 'HDV-Diesel', 或单纯 'LDV')。

        Returns:
            float: 计算出的 VSP 值，单位 $kW/tonne$。
        """
        # 1. 解析基础车型 (截取连字符前半部分)
        base_type = "LDV"  # 默认兜底保护
        if isinstance(vehicle_type, str):
            base_type = vehicle_type.split("-")[0].upper()
        
        # 2. 选取对应的物理阻力系数
        if base_type == "HDV":
            coeffs = self.hdv_coeffs
        else:
            coeffs = self.ldv_coeffs
        
        # 提取归一化阻力系数
        a_m = coeffs["a_m"]
        b_m = coeffs["b_m"]
        c_m = coeffs["c_m"]

        # ---------------------------------------------------------
        # 第一部分: 克服行驶阻力所需的功率 (Drag & Rolling Power)
        # ---------------------------------------------------------
        drag_term = a_m * v_ms + b_m * (v_ms**2) + c_m * (v_ms**3)
        
        # ---------------------------------------------------------
        # 第二部分: 克服惯性与重力所需的功率 (Inertial & Potential Power)
        # ---------------------------------------------------------
        grade_term = 9.81 * (self.road_grade / 100.0)
        inertial_term = v_ms * (1.1 * a_ms2 + grade_term)
        
        return drag_term + inertial_term
