import gi
import cv2
import numpy as np
import os

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class GstPipelineManager:
    def __init__(self, config: dict):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        Gst.init(None)

        self.video_path = config.get("video_path", "resources/test_traffic.mp4")
        self.hef_path = config.get("hef_path", "resources/yolov8m.hef")
        self.post_so_path = config.get("post_so_path", "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so")
        
        self.pipeline_string = self._build_pipeline()
        self.pipeline = Gst.parse_launch(self.pipeline_string)
        
        self.sink_video = self.pipeline.get_by_name("sink_video")
        self.sink_meta = self.pipeline.get_by_name("sink_meta")
        self.bus = self.pipeline.get_bus() 
        self.is_running = False

    def _build_pipeline(self) -> str:
        abs_path = os.path.abspath(self.video_path)
        pipeline = (
            f"filesrc location={abs_path} ! decodebin ! video/x-raw ! "
            f"videoconvert ! video/x-raw, format=NV12 ! tee name=t "
            
            # --- 视频画面分支 ---
            f"t. ! queue max-size-buffers=30 ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink name=sink_video emit-signals=false max-buffers=2 drop=false sync=false "
            
            # --- 硬件推理分支 ---
            f"t. ! queue max-size-buffers=30 ! "
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"hailonet hef-path={self.hef_path} vdevice-group-id=1 ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! "
            f"appsink name=sink_meta emit-signals=false max-buffers=2 drop=false sync=false "
        )
        return pipeline

    def start(self):
        print(f"正在启动 GStreamer 管道: 读取 {self.video_path}")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def stop(self):
        if self.is_running:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False

    def read(self):
        if not self.is_running: return None, None

        msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"\n[GStreamer 致命崩溃] {err}\n🔍 详情: {debug}\n")
            elif msg.type == Gst.MessageType.EOS:
                print(f"\n[GStreamer] 视频流已播放完毕 (EOS)\n")
            return None, None 

        sample_video = self.sink_video.emit("try-pull-sample", 500000000)
        sample_meta = self.sink_meta.emit("try-pull-sample", 500000000)
        
        if not sample_video or not sample_meta: 
            return None, None

        buffer_video = sample_video.get_buffer()
        buffer_meta = sample_meta.get_buffer()
        
        try:
            success, map_info = buffer_video.map(Gst.MapFlags.READ)
            if not success: return None, None
            
            # 动态获取视频的真实宽高
            caps = sample_video.get_caps()
            struct = caps.get_structure(0)
            actual_w = struct.get_value("width")
            actual_h = struct.get_value("height")
            
            # 使用动态获取的宽高计算预期大小
            expected_size = actual_w * actual_h * 3
            if map_info.size < expected_size:
                buffer_video.unmap(map_info)
                return None, None

            # 使用动态宽高重塑 numpy 数组
            frame = np.ndarray((actual_h, actual_w, 3), buffer=map_info.data, dtype=np.uint8).copy()
            buffer_video.unmap(map_info)

            return frame, buffer_meta
            
        except Exception as e:
            print(f">>> [GStreamer] 内存解析异常: {e}")
            return None, None
