import os
import psutil
import socket
from datetime import datetime

class SysMonitor:
    """边缘端硬件与系统状态监视器"""

    @staticmethod
    def get_system_time() -> str:
        """1. 系统时间"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def get_sd_storage() -> str:
        """边缘存储 (获取根目录 / 的磁盘使用率，通常为 SD 卡)"""
        try:
            usage = psutil.disk_usage('/')
            used_gb = usage.used / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            return f"{used_gb:.1f}G / {total_gb:.1f}G"
        except Exception:
            return "-- / --"

    @staticmethod
    def get_ssd_storage() -> str:
        """扩展存储 (获取挂载点的磁盘使用率，即 SSD)"""
        mount_point = '/app/data'
        try:
            if os.path.exists(mount_point):
                usage = psutil.disk_usage(mount_point)
                used_gb = usage.used / (1024 ** 3)
                total_gb = usage.total / (1024 ** 3)
                return f"{used_gb:.1f}G / {total_gb:.1f}G"
            else:
                return "未挂载"
        except Exception:
             return "读取失败"

    @staticmethod
    def get_network_status() -> str:
        """3. 网络连接 (优先检查 WLAN，其次检查 ETH)"""
        try:
            interfaces = psutil.net_if_addrs()
            # 树莓派常见的网卡名称
            for iface_name in ['wlan0', 'eth0']:
                if iface_name in interfaces:
                    for addr in interfaces[iface_name]:
                        # 找到 IPv4 地址
                        if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                            prefix = "WLAN" if "wlan" in iface_name else "ETH"
                            return f"{prefix} ({addr.address})"
            return "OFFLINE"
        except Exception:
            return "OFFLINE (状态未知)"

    @staticmethod
    def get_weather_gateway() -> str:
        """4. 气象网关 (假设气象站通过 USB 串口接入)"""
        # 你的气象网关可能是 RS485 转 USB。通常在树莓派上是 /dev/ttyUSB0 或 /dev/ttyACM0
        # 这里以检查 ttyUSB0 为例，如果你的网关是网络协议 (TCP/UDP)，则换成 ping 对应的 IP
        port = '/dev/ttyUSB0'
        if os.path.exists(port):
            return f"ONLINE ({port})"
        return "OFFLINE (端口未接入)"

    @staticmethod
    def get_cpu_temp() -> str:
        """5. CPU 温度 (读取树莓派底层热敏探针)"""
        try:
            # 树莓派系统中最直接且消耗最低的温度读取方式
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_c = float(f.read()) / 1000.0
            return f"{temp_c:.1f} °C"
        except Exception:
            return "-- °C"

    @staticmethod
    def get_npu_temp() -> str:
        """6. NPU 温度 (使用 Hailo 原生 Python API 获取)"""
        try:
            # 引入 Hailo 平台库
            from hailo_platform import Device
            
            # 初始化目标设备
            target = Device()
            
            # 获取芯片底层温度传感器的数据
            temp_info = target.control.get_chip_temperature()
            
            # Hailo-8 内部包含两个温度传感器 (ts0 和 ts1)
            # 我们可以直接取它们的平均值，或任意一个探针的值
            temp1 = temp_info.ts0_temperature
            temp2 = temp_info.ts1_temperature
            avg_temp = (temp1 + temp2) / 2.0
            
            return f"{avg_temp:.1f} °C"
            
        except ImportError:
            return "未检测到 hailo_platform 库"
        except Exception as e:
            # 当 NPU 正在被另一个高优先级进程（如推理引擎）独占锁定时，
            # 初始化 Device() 可能会被拒绝，此时优雅降级
            if "HAILO_OUT_OF_PHYSICAL_DEVICES" in str(e):
                return "推理中 (受进程锁保护)"
            return "OFFLINE / 未就绪"
