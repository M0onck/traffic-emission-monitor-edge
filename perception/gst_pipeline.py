import gi
import cv2
import numpy as np
import os
import logging
import infra.config.loader as cfg
from perception.math.geometry import FastUndistorter # 导入去畸变器

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logger = logging.getLogger(__name__)

def get_rpi_camera_pipeline(width=1280, height=720, fps=30):
    """
    提供给 UI 面板测试摄像头或系统降级备用的纯字符串管道。
    """
    w = int(float(width))
    h = int(float(height))
    f = int(float(fps)) 
    
    return (
        f"libcamerasrc ! "
        f"video/x-raw, format=NV12, width={w}, height={h}, framerate={f}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink name=preview_sink emit-signals=true max-buffers=2 drop=true sync=false"
    )

class GstPipelineManager:
    def __init__(self, config: dict):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        Gst.init(None)

        self.video_path = config.VIDEO_PATH
        self.hef_path = config.HEF_PATH
        self.post_so_path = config.POST_SO_PATH

        self.out_w = config.FRAME_WIDTH
        self.out_h = config.FRAME_HEIGHT
        self.use_camera = config.USE_CAMERA
        
        # 1. 初始化 Python 极速去畸变器
        self.undistorter = FastUndistorter("resources/camera_calib_6mm.npz", (self.out_w, self.out_h))

        # 2. 构建两条独立的管道
        self.src_pipeline_str, self.process_pipeline_str = self._build_pipelines()
        
        self.src_pipeline = Gst.parse_launch(self.src_pipeline_str)
        self.process_pipeline = Gst.parse_launch(self.process_pipeline_str)

        # 3. 获取所有数据交接节点
        self.raw_sink = self.src_pipeline.get_by_name("raw_sink")
        self.clean_src = self.process_pipeline.get_by_name("clean_src")
        self.meta_sink = self.process_pipeline.get_by_name("meta_sink")
        
        # 监听两条管道的总线
        self.src_bus = self.src_pipeline.get_bus() 
        self.process_bus = self.process_pipeline.get_bus()
        self.is_running = False

    def _build_pipelines(self):
        is_camera = getattr(self, 'use_camera', False) or self.video_path.startswith("libcamerasrc") or self.video_path.startswith("v4l2src")
        
        # ==================== 管道 1：源流拉取 (Camera -> Python) ====================
        if is_camera:
            source_head = f"libcamerasrc ! video/x-raw, format=NV12, width={self.out_w}, height={self.out_h}, framerate=30/1"
            sink_prop = "drop=true max-buffers=2 sync=false"
        else:
            abs_path = os.path.abspath(self.video_path)
            source_head = f"filesrc location={abs_path} ! decodebin ! video/x-raw ! videoconvert ! video/x-raw, format=NV12"
            sink_prop = "drop=false max-buffers=2 sync=false"

        src_pipeline = (
            f"{source_head} ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink name=raw_sink emit-signals=false {sink_prop}"
        )

        # ==================== 管道 2：分发处理 (Python -> AI & Record) ====================
        # appsrc 是入口，声明接收 BGR 格式
        appsrc_head = (
            f"appsrc name=clean_src is-live=true format=time ! "
            f"video/x-raw, format=BGR, width={self.out_w}, height={self.out_h}, framerate=30/1 ! "
            f"tee name=t"
        )

        record_branch = ""
        if is_camera and cfg.ENABLE_RECORD:
            os.makedirs(cfg.RECORD_SAVE_PATH, exist_ok=True)
            segment_ns = int(cfg.RECORD_SEGMENT_MIN * 60 * 1000000000)
            loc_pattern = os.path.join(cfg.RECORD_SAVE_PATH, "temp_rec_%05d.mp4")
            record_branch = (
                f" t. ! queue max-size-buffers=30 leaky=downstream ! "
                f"videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency threads=1 bitrate=2048 ! "
                f"h264parse ! splitmuxsink name=rec_sink location=\"{loc_pattern}\" max-size-time={segment_ns}"
            )

        process_pipeline = (
            f"{appsrc_head} "
            
            # --- AI 推理分支 ---
            f"t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"hailonet hef-path={self.hef_path} multi-process-service=true vdevice-group-id=SHARED ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! "
            f"appsink name=meta_sink emit-signals=false drop=true max-buffers=1 sync=false"
            
            # --- 录制分支 ---
            f"{record_branch}"
        )

        logger.info(f"构建源流管道: {src_pipeline}")
        logger.info(f"构建处理管道: {process_pipeline}")
        return src_pipeline, process_pipeline

    def start(self):
        logger.info("正在启动双核 GStreamer 管道...")
        # 必须先启动下游，再启动上游源流
        self.process_pipeline.set_state(Gst.State.PLAYING)
        self.src_pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def stop(self):
        if self.is_running:
            self.src_pipeline.set_state(Gst.State.NULL)
            
            # 给 appsrc 发送 EOS 以安全封装 mp4 录像
            self.clean_src.emit("end-of-stream")
            bus = self.process_pipeline.get_bus()
            bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS)
            
            self.process_pipeline.set_state(Gst.State.NULL)
            self.is_running = False
            logger.info("所有 GStreamer 管道已安全关闭。")

    def read(self):
        if not self.is_running: return None, None

        # 1. 从源头拉取原始畸变画面
        sample_raw = self.raw_sink.emit("try-pull-sample", 5000000) # 5ms timeout
        if not sample_raw: 
            return None, None 

        buf_raw = sample_raw.get_buffer()
        # 提取原始时间戳，后续要转交给下游，否则录制器会因时间戳混乱而崩溃
        pts = buf_raw.pts
        dts = buf_raw.dts
        duration = buf_raw.duration

        try:
            success, map_info = buf_raw.map(Gst.MapFlags.READ)
            if not success: return None, None
            raw_frame = np.ndarray((self.out_h, self.out_w, 3), buffer=map_info.data, dtype=np.uint8)
            
            # 2. 调用 Python 硬件加速去畸变 (耗时 ~3-5ms)
            clean_frame = self.undistorter.process(raw_frame)
            
        finally:
            buf_raw.unmap(map_info)

        # 3. 将干净的画面重新封装为 GstBuffer，推入处理管道 (AI / Record)
        clean_data = clean_frame.tobytes()
        buf_clean = Gst.Buffer.new_wrapped(clean_data)
        buf_clean.pts = pts
        buf_clean.dts = dts
        buf_clean.duration = duration
        
        # 注入管道
        self.clean_src.emit("push-buffer", buf_clean)

        # 4. 从处理管道拉取 AI 推理结果
        sample_meta = self.meta_sink.emit("try-pull-sample", 1000000)
        buffer_meta = sample_meta.get_buffer() if sample_meta else Gst.Buffer.new()

        # 将干净的画面返回给 UI 显示
        return clean_frame, buffer_meta

    def set_record_location(self, session_id):
        rec_sink = self.process_pipeline.get_by_name("rec_sink")
        if rec_sink:
            import time
            timestamp = int(time.time())
            filename = f"{session_id}_seq%05d_start{timestamp}.mp4"
            full_path = os.path.join(cfg.RECORD_SAVE_PATH, filename)
            rec_sink.set_property("location", full_path)
            logger.info(f"录制路径已更新为: {full_path}")
