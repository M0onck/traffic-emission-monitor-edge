#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional

class RawEnvSanitizer:
    """
    环境原始数据净化器 (Raw Environment Data Sanitizer)
    专门拦截和修复传感器硬件毛刺，匹配真实入库键名。
    """
    def __init__(self):
        # 传感器数值物理边界过滤
        self._bounds = {
            'pm25_raw': (0.0, 2000.0),       # ug/m3
            'pm10_raw': (0.0, 3000.0),       # ug/m3
            'air_temp': (-40.0, 60.0),       # 气温 °C
            'humidity': (0.0, 100.0),        # 相对湿度 %
            'wind_speed': (0.0, 50.0),       # 风速 m/s
            'wind_dir': (0.0, 360.0),        # 风向度数
            'ground_temp': (-40.0, 100.0)    # 路面/地面温度 °C
        }
        
        # 允许的最大单秒跳变率 (防止高频毛刺)
        self._max_roc = {
            'air_temp': 5.0, 
            'humidity': 10.0,
            'ground_temp': 10.0
        }
        
        self._last_valid_state: Dict[str, float] = {}

    def sanitize(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤并插补异常值
        """
        if not raw_data:
            return {}

        sanitized = raw_data.copy()

        for key, value in sanitized.items():
            if key not in self._bounds:
                continue
                
            is_valid = True
            
            if value is None:
                is_valid = False
            else:
                try:
                    val_float = float(value)
                    # 1. 物理边界检查
                    min_v, max_v = self._bounds[key]
                    if not (min_v <= val_float <= max_v):
                        is_valid = False
                        
                    # 2. 单秒突变率检查
                    elif key in self._max_roc and key in self._last_valid_state:
                        last_v = self._last_valid_state[key]
                        if abs(val_float - last_v) > self._max_roc[key]:
                            is_valid = False
                            
                except (ValueError, TypeError):
                    is_valid = False

            if is_valid:
                self._last_valid_state[key] = val_float
                sanitized[key] = val_float
            else:
                # 异常处理：前向插补或置空
                sanitized[key] = self._last_valid_state.get(key, None)

        return sanitized
