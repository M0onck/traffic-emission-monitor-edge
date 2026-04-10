import gi
import cv2
import numpy as np
import os
import infra.config.loader as cfg

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def get_rpi_camera_pipeline(width=1440, height=1080, fps=30):
    """纯画面拉流管道 (原生 GStreamer 专用)"""
    # 强制转换 fps 为 int，防止出现 30.0/1 导致 GStreamer 解析为 string 类型
    fps_int = int(fps)
    return (
        f"libcamerasrc ! "
        f"video/x-raw, format=NV12, width={width}, height={height}, framerate={fps_int}/1 ! "
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
        
        # 实时拉流和离线文件分支构建逻辑
        if is_camera:
            source_head = f"libcamerasrc ! video/x-raw, format=NV12, width={self.out_w}, height={self.out_h}, framerate=30/1"
            # 实时拉流：必须开启丢帧策略，保证永远获取最新画面，实现零延迟
            queue_prop = "leaky=downstream"
            sink_prop = "drop=true sync=false"
        else:
            abs_path = os.path.abspath(self.video_path)
            source_head = f"filesrc location={abs_path} ! decodebin ! video/x-raw ! videoconvert ! video/x-raw, format=NV12"
            # 离线文件：关闭所有丢帧属性。利用 Python 端循环速度反向阻塞 (Backpressure) GStreamer 解码流速
            queue_prop = "" 
            sink_prop = "drop=false sync=false"

        # 录制分支构建逻辑
        record_branch = ""
        # 仅在物理摄像头模式且打开了录制开关时，激活录制管道
        if is_camera and cfg.ENABLE_RECORD:
            os.makedirs(cfg.RECORD_SAVE_PATH, exist_ok=True)
            
            # splitmuxsink 的 max-size-time 单位是纳秒 (ns)
            segment_ns = int(cfg.RECORD_SEGMENT_MIN * 60 * 1000000000)
            
            # 文件命名模板：record_00001.mp4, record_00002.mp4...
            loc_pattern = os.path.join(cfg.RECORD_SAVE_PATH, "record_%05d.mp4")
            
            # 采用高兼容性软件编码器方案：
            # 1. 增加 videoconvert 自动协商色彩空间（x264enc 通常偏好 I420）
            # 2. 使用 x264enc 替代 v4l2h264enc，并使用 ultrafast 降低 CPU 占用
            record_branch = (
                f"t. ! queue max-size-buffers=30 leaky=downstream ! "
                f"videoconvert ! "
                f"x264enc speed-preset=ultrafast tune=zerolatency threads=4 bitrate=2048 ! "
                f"h264parse ! splitmuxsink location=\"{loc_pattern}\" max-size-time={segment_ns} "
            )

        pipeline = (
            f"{source_head} ! tee name=t "
            
            # --- 分支1: 视频画面分支 ---
            f"t. ! queue max-size-buffers=2 {queue_prop} ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink name=sink_video emit-signals=false max-buffers=1 {sink_prop} "
            
            # --- 分支2: AI 推理分支 ---
            f"t. ! queue max-size-buffers=2 {queue_prop} ! "
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"hailonet hef-path={self.hef_path} ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! "
            f"appsink name=sink_meta emit-signals=false max-buffers=1 {sink_prop} "

            # --- 分支3: 录制切片分支 ---
            f"{record_branch}"
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

    def read(self):
        if not self.is_running: return None, None

        msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"\n[GStreamer 致命崩溃] {err}\n详情: {debug}\n")
                self.is_running = False
                
            elif msg.type == Gst.MessageType.EOS:
                print(f"\n[GStreamer] 视频流已播放完毕 (EOS)\n")
                self.is_running = False
                
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
        
        finally:
            # 无论前面发生什么崩溃，finally 保证一定会释放视频画面内存锁
            buffer_video.unmap(map_info)
