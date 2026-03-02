import gi
import cv2
import numpy as np
import time
import os

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class GstPipelineManager:
    """
    [感知层] GStreamer 管道管理器 (双路提取·绝对防崩溃版)
    功能：构建并管理底层硬件加速流水线。
    数据流向：视频解码 -> 分流 -> (1080p图像, 640p带元数据Buffer) -> Python提取
    """
    def __init__(self, config: dict):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        Gst.init(None)

        self.video_path = config.get("video_path", "resources/test_traffic.mp4")
        self.hef_path = config.get("hef_path", "resources/yolov8m.hef")
        self.post_so_path = config.get("post_so_path", "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so")
        
        self.pipeline_string = self._build_pipeline()
        self.pipeline = Gst.parse_launch(self.pipeline_string)
        
        # 核心魔法：获取两个提取端口
        self.sink_video = self.pipeline.get_by_name("sink_video")
        self.sink_meta = self.pipeline.get_by_name("sink_meta")
        
        self.is_running = False

    def _build_pipeline(self) -> str:
        """
        【终极解耦架构】彻底抛弃不稳定的 hailoaggregator C++ 插件！
        """
        abs_path = os.path.abspath(self.video_path)
        
        pipeline = (
            f"filesrc location={abs_path} ! decodebin ! video/x-raw ! "
            f"videoconvert ! video/x-raw, format=NV12 ! "
            f"videoscale ! video/x-raw, width=1920, height=1080 ! tee name=t "
            
            # ================= 分支 1：高分辨率原画直出 =================
            # drop=false 保证 GStreamer 绝对不丢帧，等待 Python 同步拉取
            f"t. ! queue max-size-buffers=30 leaky=no ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink name=sink_video emit-signals=false max-buffers=30 drop=false sync=false "
            
            # ================= 分支 2：AI 推理输出元数据 =================
            f"t. ! queue max-size-buffers=30 leaky=no ! "
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"hailonet hef-path={self.hef_path} vdevice-group-id=1 ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! "
            f"hailotracker name=hailo_tracker keep-tracked-frames=3 class-id=-1 ! "
            f"appsink name=sink_meta emit-signals=false max-buffers=30 drop=false sync=false "
        )
        return pipeline

    def start(self):
        print(f"🚀 正在启动 GStreamer 管道: 读取 {self.video_path}")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def stop(self):
        if self.is_running:
            print("🛑 正在停止管道...")
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False

    def read(self):
        if not self.is_running:
            return None, None

        # ================= 完美同步提取 =================
        # 由于我们设置了 drop=false，这里拉取到的图像和元数据必定属于同一微秒的同一帧！
        sample_video = self.sink_video.emit("pull-sample")
        sample_meta = self.sink_meta.emit("pull-sample")
        
        if not sample_video or not sample_meta:
            return None, None

        buffer_video = sample_video.get_buffer()
        buffer_meta = sample_meta.get_buffer()
        
        try:
            # 1. 提取 1080p 的干净画面
            success, map_info = buffer_video.map(Gst.MapFlags.READ)
            if not success:
                return None, None
                
            expected_size = 1920 * 1080 * 3
            if map_info.size < expected_size:
                buffer_video.unmap(map_info)
                return None, None

            frame = np.ndarray(
                (1080, 1920, 3), 
                buffer=map_info.data, 
                dtype=np.uint8
            ).copy()
            
            buffer_video.unmap(map_info)
            
            # 2. 将 1080p 的画面，加上携带物理追踪结果的 640p 缓冲，一并返回！
            # monitor_engine.py 的 hailo 解析器会极其聪明地根据相对比例读取框的坐标。
            return frame, buffer_meta
            
        except Exception as e:
            print(f">>> [GStreamer] 内存解析异常: {e}")
            return None, None
