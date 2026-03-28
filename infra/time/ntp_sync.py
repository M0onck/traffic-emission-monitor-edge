import time
import threading
import logging

try:
    import ntplib
except ImportError:
    ntplib = None

class TimeSynchronizer:
    """
    [基础设施] 核心时钟同步器
    负责与远端 NTP 服务器保持对齐，计算本地时钟的系统漂移 (Offset)。
    即使网络断开，也能利用最后一次准确的 Offset 进行相对准确的时间推算。
    """
    def __init__(self, ntp_server="pool.ntp.org", sync_interval_sec=3600):
        self.ntp_server = ntp_server
        self.sync_interval = sync_interval_sec
        self.offset = 0.0
        self.is_synced = False
        self._stop_event = threading.Event()
        
        # 启动时进行首次阻塞式同步，确保初始时间正确
        self.sync_now()
        
        # 开启后台守护线程，定期校准时钟漂移
        self.sync_thread = threading.Thread(target=self._background_routine, daemon=True)
        self.sync_thread.start()

    def sync_now(self) -> bool:
        if ntplib is None:
            logging.error("[TimeSync] ntplib 未安装，降级使用本地系统时间。")
            return False
            
        try:
            client = ntplib.NTPClient()
            # 设置较短的超时时间，防止阻塞主线程太久
            response = client.request(self.ntp_server, version=3, timeout=3.0)
            self.offset = response.offset
            self.is_synced = True
            logging.info(f"[TimeSync] NTP 同步成功。本地时钟偏移量: {self.offset:.6f} 秒")
            return True
        except Exception as e:
            logging.warning(f"[TimeSync] NTP 同步失败: {e}。将维持现有 Offset: {self.offset:.6f} 秒")
            return False

    def _background_routine(self):
        while not self._stop_event.is_set():
            # 等待设定的同步间隔
            self._stop_event.wait(self.sync_interval)
            if not self._stop_event.is_set():
                self.sync_now()

    def get_precise_timestamp(self) -> float:
        """
        [核心接口] 获取高精度时间戳
        供视频帧流高频调用。返回补齐了 NTP 偏移量的绝对 UNIX 时间戳。
        """
        # time.time() 获取本地时间，加上网络同步算出的 offset
        return time.time() + self.offset
        
    def stop(self):
        """安全释放后台线程"""
        self._stop_event.set()
