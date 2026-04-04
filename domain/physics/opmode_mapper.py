class OpModeMapper:
    """
    [物理层/业务层] MOVES 宏观工况映射与时序分析器
    功能：接收离场车辆的完整平滑轨迹，应用持续时间阈值滤波，提取主导工况。
    """
    def __init__(self, duration_threshold: float = 1.0):
        # 持续时间阈值（秒），过滤高频抖动导致的“伪工况”
        self.duration_threshold = duration_threshold

    @staticmethod
    def get_instant_opmode(v_ms: float, vsp: float) -> str:
        """瞬时物理状态到工况的映射 (参考 EPA MOVES)"""
        if v_ms < 1.0:
            return "Idling"       # 怠速
        if vsp < 0:
            return "Braking"      # 减速/刹车
        elif 0 <= vsp < 12.0:
            return "Cruise"       # 巡航
        else:
            return "Acceleration" # 加速

    def extract_dominant_opmodes(self, trajectory_data: list) -> list:
        """
        批处理整段轨迹，提取符合时间阈值的主导工况。
        
        :param trajectory_data: 包含 [{'timestamp': float, 'v_ms': float, 'vsp': float}, ...] 的列表
        :return: 按有效持续时间降序排列的工况名称列表，如 ["Cruise", "Acceleration"]
        """
        if not trajectory_data:
            return ["Unknown"]

        valid_opmodes_time = {}
        
        # 初始化第一帧状态
        first_pt = trajectory_data[0]
        current_opmode = self.get_instant_opmode(first_pt['v_ms'], first_pt['vsp'])
        segment_start_time = first_pt['timestamp']
        last_time = first_pt['timestamp']

        for pt in trajectory_data[1:]:
            t = pt['timestamp']
            v = pt['v_ms']
            vsp = pt['vsp']
            
            instant_opmode = self.get_instant_opmode(v, vsp)

            # 发生工况切换
            if instant_opmode != current_opmode:
                duration = last_time - segment_start_time
                
                # 只有该段工况持续时间超过阈值，才视为有效并累加
                if duration >= self.duration_threshold:
                    valid_opmodes_time[current_opmode] = valid_opmodes_time.get(current_opmode, 0.0) + duration
                
                # 重置新工况段的起点
                current_opmode = instant_opmode
                segment_start_time = t
                
            last_time = t

        # 循环结束，不要忘记结算最后一段未闭合的工况
        final_duration = last_time - segment_start_time
        if final_duration >= self.duration_threshold:
            valid_opmodes_time[current_opmode] = valid_opmodes_time.get(current_opmode, 0.0) + final_duration

        # 如果轨迹太短或者频繁抖动导致没有工况能超过 1 秒
        if not valid_opmodes_time:
            return ["Transient"]

        # 按累计有效时间从大到小排序并提取名称
        sorted_modes = sorted(valid_opmodes_time.items(), key=lambda x: x[1], reverse=True)
        return [mode[0] for mode in sorted_modes]
