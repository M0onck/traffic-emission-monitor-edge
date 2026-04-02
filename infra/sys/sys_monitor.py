import os
import psutil
import socket
from datetime import datetime
import subprocess

class SysMonitor:
    """边缘端硬件与系统状态监视器"""

    @staticmethod
    def get_system_time() -> str:
        """1. 系统时间"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def get_edge_storage() -> str:
        """2. 边缘存储 (获取根目录 / 的磁盘使用率)"""
        try:
            usage = psutil.disk_usage('/')
            used_gb = usage.used / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            return f"{used_gb:.1f} GB / {total_gb:.1f} GB"
        except Exception:
            return "-- GB / -- GB"

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
        """6. NPU 温度 (以 Hailo-8 为例)"""
        try:
            # 方式A: 通过 Hailo 官方命令行工具获取 (推荐)
            # 命令大概输出: Temperature: 45.26 Degrees
            result = subprocess.check_output(
                ['hailortcli', 'fw-control', 'measure-temp'], 
                text=True, stderr=subprocess.DEVNULL
            )
            # 提取数字部分
            import re
            match = re.search(r'([\d\.]+)', result)
            if match:
                return f"{float(match.group(1)):.1f} °C"
            return "运行中 (温度暂缺)"
        except Exception:
            # 如果没装 CLI 或者板子没插好
            return "OFFLINE / 未就绪"
