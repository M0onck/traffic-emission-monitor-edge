# domain/physics/alignment_engine.py
import json
import logging
import time
from typing import Dict, Any, List
import infra.config.loader as cfg

logger = logging.getLogger(__name__)

class AlignmentEngine:
    """
    [业务层] 延迟对齐引擎 (1Hz 快照版)
    职责: 定期扫描数据库，以 T_now - Delay 为基准时间点，
          横向切片式地提取该时刻的环境数据与所有存活车辆的瞬时运动学状态。
    """
    def __init__(self, db_manager):
        self.db = db_manager
        
        # 1. 解析时间对齐参数
        self.delay_sec = getattr(cfg, 'ALIGNMENT_DELAY_SEC', 60.0)
        self.max_env_tolerance = 2.0  # 环境数据匹配的最大容差 (秒)

    def process_alignment_tick(self, session_id: str, current_system_time: float) -> Dict[str, Any]:
        """
        执行单次 1Hz 对齐拨号，产出该时刻的物理世界快照
        """
        # 计算当前需要对齐的历史锚点时间戳
        target_time = current_system_time - self.delay_sec
        
        # 1. 提取该时刻的环境数据快照 (取最接近 target_time 的一条记录)
        env_data = self.db.get_nearest_env_raw(session_id, target_time, self.max_env_tolerance)
        
        # 2. 提取该时刻的场内车辆运动学快照
        vehicles_snapshot = self._extract_vehicles_at_time(session_id, target_time)
        
        if not vehicles_snapshot and not env_data:
            return None

        # 3. 构造标准的 L2 级对齐快照数据结构
        snapshot = {
            'session_id': session_id,
            'aligned_timestamp': target_time,
            'air_temp': env_data.get('air_temp') if env_data else None,
            'ground_temp': env_data.get('ground_temp') if env_data else None,
            'humidity': env_data.get('humidity') if env_data else None,
            'wind_speed': env_data.get('wind_speed') if env_data else None,
            'wind_dir': env_data.get('wind_dir') if env_data else None,
            'pm25': env_data.get('pm25_raw') if env_data else None,
            'pm10': env_data.get('pm10_raw') if env_data else None,
            'active_vehicle_count': len(vehicles_snapshot),
            'vehicles_data': json.dumps(vehicles_snapshot) # 序列化为 JSON 存储
        }
        
        logger.debug(f"[Alignment] 已生成 T={target_time:.1f} 的物理快照，包含 {len(vehicles_snapshot)} 辆车")
        return snapshot

    def _extract_vehicles_at_time(self, session_id: str, target_time: float) -> List[Dict[str, Any]]:
        """
        根据时间戳，从 Veh_Raw 表的轨迹 BLOB 中捞出该时刻在场的所有车辆及其瞬时状态
        """
        # 从数据库中获取所有在 target_time 期间处于场内的车辆记录
        active_rows = self.db.get_active_vehicles_during(session_id, target_time)
        
        snapshot_list = []
        for row in active_rows:
            tid = row['tracker_id']
            v_type = row['vehicle_type']
            
            try:
                # 解析轨迹 JSON (格式: [{"timestamp":..., "x":..., "v":...}, ...])
                trajectory = json.loads(row['trajectory_blob'])
                
                # 寻找离 target_time 最近的一个轨迹点
                # 由于轨迹是以 5Hz 记录的，最近点的误差通常在 100ms 以内
                nearest_pt = min(trajectory, key=lambda p: abs(p['timestamp'] - target_time))
                
                # 容差过滤：如果最近的点离目标时间超过 1 秒，说明存在数据断层，不计入快照
                if abs(nearest_pt['timestamp'] - target_time) > 1.0:
                    continue
                
                snapshot_list.append({
                    'tid': tid,
                    'type': v_type,
                    'x': round(nearest_pt.get('x', 0.0), 3),
                    'y': round(nearest_pt.get('y', 0.0), 3),
                    'v': round(nearest_pt.get('v', 0.0), 3),
                    'a': round(nearest_pt.get('a', 0.0), 3),
                    'vsp': round(nearest_pt.get('vsp', 0.0), 3)
                })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"[Alignment] 解析 TID={tid} 的轨迹数据失败: {e}")
                continue
                
        return snapshot_list
