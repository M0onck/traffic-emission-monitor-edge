import gi
import cv2
import numpy as np
import os
import infra.config.loader as cfg

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def get_rpi_camera_pipeline(width=1440, height=1080, fps=30):
    """纯画面拉流管道 (原生 GStreamer 专用)"""
    return (
        f"libcamerasrc ! "
        f"video/x-raw, format=NV12, width={width}, height={height}, framerate={fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        # 指定名字为 preview_sink，并允许 emit 提取画面
        f"appsink name=preview_sink emit-signals=true max-buffers=2 drop=true sync=false"
    )

class GstPipelineManager:
    def __init__(self, config: dict):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        Gst.init(None)

        self.video_path = config.get("video_path", "resources/test_traffic.mp4")
        self.hef_path = config.get("hef_path", "resources/yolov8s.hef")
        self.post_so_path = config.get("post_so_path", "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so")

        self.out_w = config.get("FRAME_WIDTH", 1280) 
        self.out_h = config.get("FRAME_HEIGHT", 720)
        
        self.use_camera = cfg.USE_CAMERA
        self.pipeline_string = self._build_pipeline()
        self.pipeline = Gst.parse_launch(self.pipeline_string)
        
        self.sink_video = self.pipeline.get_by_name("sink_video")
        self.sink_meta = self.pipeline.get_by_name("sink_meta")
        self.bus = self.pipeline.get_bus() 
        self.is_running = False
        self._kept_sample = None

    def _build_pipeline(self) -> str:
        is_camera = getattr(self, 'use_camera', False) or self.video_path.startswith("libcamerasrc") or self.video_path.startswith("v4l2src")
        
        if is_camera:
            source_head = f"libcamerasrc ! video/x-raw, format=NV12, width={self.out_w}, height={self.out_h}, framerate=30/1"
        else:
            abs_path = os.path.abspath(self.video_path)
            source_head = f"filesrc location={abs_path} ! decodebin ! video/x-raw ! videoconvert ! video/x-raw, format=NV12"

        pipeline = (
            f"{source_head} ! tee name=t "
            
            # --- 分支1: 视频画面分支 ---
            # 队列限制为 2，并且开启 leaky=downstream（满了就丢弃旧帧）
            f"t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink name=sink_video emit-signals=false max-buffers=1 drop=true sync=false "
            
            # --- 分支2: AI 推理分支 ---
            f"t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"hailonet hef-path={self.hef_path} ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! "
            f"appsink name=sink_meta emit-signals=false max-buffers=1 drop=true sync=false "
        )
        return pipeline

    def start(self):
        print(f"正在启动 GStreamer 推理管道...")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def stop(self):
        if self.is_running:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False

    # 替换整个 read 方法：
    def read(self):
        if not self.is_running: return None, None

        msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"\n[GStreamer 致命崩溃] {err}\n详情: {debug}\n")
            elif msg.type == Gst.MessageType.EOS:
                print(f"\n[GStreamer] 视频流已播放完毕 (EOS)\n")
            return None, None 

        # 1. 获取视频帧 (允许5ms超时)
        sample_video = self.sink_video.emit("try-pull-sample", 5000000)
        if not sample_video: 
            return None, None 

        # 2. 尝试获取 AI 结果 (仅等待1ms)
        sample_meta = self.sink_meta.emit("try-pull-sample", 1000000)

        if sample_meta:
            buffer_meta = sample_meta.get_buffer()
            # 拿到 buffer_meta 后，sample_meta 局部变量会在函数结束时被 Python 垃圾回收
            # 底层 C++ 内存会立刻被无缝释放回 GStreamer 内存池，管道永远畅通！
        else:
            # 防丢帧欺骗策略
            # 如果 AI 没赶上，造一个空的假 Buffer 骗过 monitor_engine.py
            # 这样引擎就不会触发 "if buffer is None: continue" 把视频帧丢掉了，黑屏彻底解决！
            buffer_meta = Gst.Buffer.new()

        buffer_video = sample_video.get_buffer()
        try:
            success, map_info = buffer_video.map(Gst.MapFlags.READ)
            if not success: return None, None
            
            caps = sample_video.get_caps()
            struct = caps.get_structure(0)
            actual_w = struct.get_value("width")
            actual_h = struct.get_value("height")
            
            frame = np.ndarray((actual_h, actual_w, 3), buffer=map_info.data, dtype=np.uint8).copy()
            buffer_video.unmap(map_info)

            return frame, buffer_meta
            
        except Exception as e:
            print(f">>> [GStreamer] 内存解析异常: {e}")
            return None, None
