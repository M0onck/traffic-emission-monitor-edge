import ctypes
import os
import time

class WeatherDataC(ctypes.Structure):
    """必须与 C++ 中的 struct 内存布局完全一致"""
    _fields_ = [
        ("temp", ctypes.c_float),
        ("humidity", ctypes.c_float),
        ("windSpeed", ctypes.c_float),
        ("windDir", ctypes.c_int),
        ("pm25", ctypes.c_int),
        ("pm10", ctypes.c_int),
        ("timestamp", ctypes.c_uint32),
        ("isOnline", ctypes.c_bool)
    ]

class WeatherGateway:
    def __init__(self, lib_path="lib/libweather.so"):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"找不到气象站驱动库: {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # 定义 C 函数的返回类型
        self.lib.get_weather_data.restype = WeatherDataC
        self.lib.send_sync_cmd.argtypes = [ctypes.c_uint32]
        
    def start(self):
        self.lib.start_monitor()
        
    def stop(self):
        self.lib.stop_monitor()
        
    def get_data(self) -> dict:
        """获取并转换为 Python 字典"""
        data = self.lib.get_weather_data()
        return {
            "temp": data.temp,
            "humidity": data.humidity,
            "windSpeed": data.windSpeed,
            "windDir": data.windDir,
            "pm25": data.pm25,
            "pm10": data.pm10,
            "timestamp": data.timestamp,
            "isOnline": data.isOnline
        }
        
    def sync_time(self):
        """下发当前系统的绝对时间戳"""
        current_ts = int(time.time())
        self.lib.send_sync_cmd(current_ts)
        
    def zero_wind(self):
        """下发风速调零指令"""
        self.lib.send_zero_cmd()
