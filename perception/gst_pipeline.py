import gi
import numpy as np
import os
import time
from datetime import datetime
import logging
import infra.config.loader as cfg

# ==========================================
# 3. GStreamer 初始化
# ==========================================
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
    def __init__(self, config: dict, shm_array=None, frame_ready_event=None, force_no_record=False, progress_callback=None):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        Gst.init(None)
        self.config = config
        self._shm_array = shm_array  # 直接持有共享内存引用
        self._frame_ready_event = frame_ready_event
        self.video_path = config.VIDEO_PATH
        self.hef_path = config.HEF_PATH
        self.out_w = config.FRAME_WIDTH
        self.out_h = config.FRAME_HEIGHT
        self.use_camera = config.USE_CAMERA
        self.force_no_record = force_no_record
        self.session_id = getattr(config, 'CURRENT_SESSION_ID', 'debug_session')
        self.progress_callback = progress_callback

        # === 在构建管线前报告进度 ===
        if self.progress_callback:
            self.progress_callback(30, "正在构建 GPU 加速的视频处理管线...")

        # 构建底层功能齐全（去畸变、录像、AI、预览）的并行大管道
        self.pipeline_str = self._build_pipelines()
        self.pipeline = Gst.parse_launch(self.pipeline_str)

        # 绑定动态命名回调
        rec_sink = self.pipeline.get_by_name("rec_sink")
        if rec_sink:
            rec_sink.connect("format-location", self._on_format_location)

        # 配置 clean_sink 的信号模式
        self.clean_sink = self.pipeline.get_by_name("clean_sink")
        if self.clean_sink:
            self.clean_sink.set_property("emit-signals", True)
            # 关键：当底层 GStreamer 线程收到新帧，立刻触发 _on_new_sample
            self.clean_sink.connect("new-sample", self._on_new_sample)
        
        # 必须作为类的全局属性进行初始化，抵抗 Python 惰性 GC 的销毁
        self.meta_sink = None 
        
        self.is_running = False
        self.last_hailo_data = []
        self._latest_frame = None

        self.ai_frame = None # 用于缓存给 NPU 的最新帧
        ai_sink = self.pipeline.get_by_name("ai_sink")
        if ai_sink:
            ai_sink.connect("new-sample", self._on_ai_sample)

    def _on_format_location(self, _splitmux, fragment_id):
        """
        每次录制切片时由 GStreamer 回调，动态生成含有精确当前时间戳的文件名
        """
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_id}_seq{fragment_id:05d}_start{current_time_str}.mkv"
        return os.path.join(self.config.RECORD_SAVE_PATH, filename)

    def _on_ai_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample: return gi.repository.Gst.FlowReturn.OK
        try:
            buffer = sample.get_buffer()
            success, map_info = buffer.map(gi.repository.Gst.MapFlags.READ)
            if success:
                try:
                    frame = np.ndarray(shape=(640, 640, 3), dtype=np.uint8, buffer=map_info.data)
                    self.ai_frame = frame.copy() # 拷贝一帧用于 AI
                    if self._frame_ready_event:
                        self._frame_ready_event.set() # 敲响唤醒铜锣
                finally:
                    buffer.unmap(map_info)
        finally:
            sample = None
        return gi.repository.Gst.FlowReturn.OK

    def _on_new_sample(self, sink):
        """
        [极速回调] 由 GStreamer 内部线程直接触发。
        """
        sample = sink.emit("pull-sample")
        if not sample:
            return gi.repository.Gst.FlowReturn.OK

        try:
            buffer = sample.get_buffer()
            success, map_info = buffer.map(gi.repository.Gst.MapFlags.READ)
            if success:
                try:
                    # 尝试映射并拷贝
                    frame_view = np.ndarray(
                        shape=(self.out_h, self.out_w, 3),
                        dtype=np.uint8,
                        buffer=map_info.data
                    )
                    
                    if self._shm_array is not None:
                        # [模式 A] 守护进程：极速穿透写共享内存
                        np.copyto(self._shm_array, frame_view)
                    else:
                        # [模式 B] UI 标定：深拷贝保存至本地缓存
                        self._latest_frame = frame_view.copy()
                        
                except Exception as e:
                    # 万一拷贝发生异常，记录下来，绝不能让程序静默死锁
                    logger.error(f"[GstPipeline] 内存搬运回调发生异常: {e}")
                finally:
                    # 只要 map 成功了，无论中间报什么错，都必须在此 unmap 归还内存！
                    buffer.unmap(map_info)
        finally:
            # 销毁 sample，触发 GStreamer 内部的 unref
            sample = None
            
        return gi.repository.Gst.FlowReturn.OK

    def _build_pipelines(self):
        """
        构建工业级边缘计算流媒体管线。
        """
        is_camera = getattr(self, 'use_camera', False) or self.video_path.startswith("libcamerasrc") or self.video_path.startswith("v4l2src")
        
        base_dir = os.getcwd() 
        map_bin = os.path.join(base_dir, "resources", "dewarp_map_rgba_1456x1088.bin")

        cam_w, cam_h = 1456, 1088
        out_w, out_h = self.out_w, self.out_h

        # 如果强制不录像，或者全局开关没开，则不构建录像分支
        enable_rec = getattr(self.config, 'ENABLE_RECORD', False)
        if self.force_no_record:
            enable_rec = False
        
        # --- 1. 录像分支 ---
        record_branch = ""
        if is_camera and enable_rec:
            os.makedirs(cfg.RECORD_SAVE_PATH, exist_ok=True)
            segment_ns = int(cfg.RECORD_SEGMENT_MIN * 60 * 1000000000)

            record_branch = (
                f" t. ! queue max-size-buffers=60 ! "
                f"videorate ! "  
                f"videoconvert ! video/x-raw,format=I420 ! "
                f"x264enc speed-preset=ultrafast tune=zerolatency threads=1 bitrate=2048 key-int-max=60 ! "
                f"h264parse config-interval=1 ! "
                f"splitmuxsink name=rec_sink muxer=matroskamux max-size-time={segment_ns} "
            )

        # --- 2. 源分支（切换摄像头或者视频文件） ---
        if is_camera:
            source_head = (
                f"tcpclientsrc host=127.0.0.1 port=5000 do-timestamp=true ! "
                f"jpegparse ! "
                f"queue max-size-buffers=1 leaky=downstream ! " 
                f"jpegdec ! "
                f"videoconvert ! video/x-raw, format=RGBA"
            )
        else:
            abs_path = os.path.abspath(self.video_path)
            source_head = f"filesrc location={abs_path} ! decodebin ! video/x-raw ! videoconvert ! video/x-raw, format=NV12"

        crop_left = (cam_w - out_w) // 2
        crop_right = cam_w - out_w - crop_left
        crop_top = (cam_h - out_h) // 2
        crop_bottom = cam_h - out_h - crop_top

        # --- 3. GPU 去畸变主干 ---
        source_section = (
            f"{source_head} ! "
            f"glupload ! glcolorconvert ! "
            f"dewarpfilter map-file-path={map_bin} map-width={cam_w} map-height={cam_h} ! " 
            f"glcolorconvert ! gldownload ! videoconvert ! "
            f"videocrop top={crop_top} bottom={crop_bottom} left={crop_left} right={crop_right} ! " 
            f"video/x-raw, width={out_w}, height={out_h}, format=BGR ! tee name=t " 
        )

        # --- 4. AI 推理分支 ---
        ai_branch = (
            f" t. ! queue name=ai_q max-size-buffers=2 leaky=downstream ! "
            f"videoscale qos=false ! video/x-raw, width=640, height=640 ! "
            f"videoconvert qos=false ! video/x-raw, format=RGB ! "
            f"appsink name=ai_sink emit-signals=true max-buffers=1 drop=true sync=false async=false "
        )

        # --- 5. UI 渲染业务提取分支 ---
        preview_branch = (
            f" t. ! queue name=ui_q max-size-buffers=1 leaky=downstream ! "
            f"appsink name=clean_sink emit-signals=false max-buffers=1 drop=true sync=false async=false "
        )

        final_pipeline_str = f"{source_section}{ai_branch}{preview_branch}{record_branch}"
        logger.info(f"构建 GPU 加速的边缘重构管道: {final_pipeline_str}")
        return final_pipeline_str

    def start(self):
        # === 在启动引擎前报告进度 ===
        if self.progress_callback:
            self.progress_callback(60, "正在启动 GStreamer 硬件数据流引擎...")
        logger.info("正在启动 GStreamer 管道 (原生 PyHailoRT 抓图模式)...")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def wait_for_first_frame(self, timeout=10.0) -> bool:
        """
        阻塞等待硬件传感器预热并吐出第一帧有效画面。
        这能杜绝进入 UI 标定界面时出现“黑屏闪烁”的问题。
        """
        if self.progress_callback:
            self.progress_callback(85, "正在拉取并解析首帧预处理画面...")
            
        start_time = time.time()
        # 循环检测缓存帧是否已经被 _on_new_sample 填充
        while self._latest_frame is None:
            if time.time() - start_time > timeout:
                logger.error("[GstPipeline] 获取首帧超时，硬件可能离线或被占用！")
                return False
            time.sleep(0.1) # 100ms 轮询一次，不阻塞 CPU

        # === 在系统就绪时报告进度 ===    
        if self.progress_callback:
            self.progress_callback(100, "系统就绪，视觉通道已建立...")
            
        return True

    def stop(self):
        """安全停止流水线。顺序极其严格，防止底层撕裂。"""
        if self.is_running:

            # 发送 EOS 信号：保证所有缓冲数据落盘，mp4 文件拥有合法的 moov 尾部头
            self.pipeline.send_event(Gst.Event.new_eos())
            
            # 总线等待：给予系统最多 12 秒处理完结帧
            bus = self.pipeline.get_bus()
            if bus:
                # 同时监听 EOS 和 ERROR
                # 这样如果底层 muxer 发生物理性写入失败，可以立即在终端看到报错
                msg = bus.timed_pop_filtered(
                    12 * Gst.SECOND, 
                    Gst.MessageType.EOS | Gst.MessageType.ERROR
                )
                
                if msg and msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    logger.error(f"[GStreamer] 视频封装时发生底层致命错误: {err.message}\n调试信息: {debug}")
                
            # 彻底物理释放
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
            logger.info("GStreamer 流水线已安全关闭，录像封装完毕。")

    def read(self):
        """
        [向下兼容接口] 供主进程 UI 标定界面使用。
        返回最新缓存的画面和 AI 元数据。
        注意：在多进程引擎正式启动后，系统不应再调用此接口，而是通过共享内存读取。
        """
        # 如果还没收到第一帧，返回 None
        if self._latest_frame is None:
            return None
            
        return self._latest_frame

    def release_frame(self):
        """
        由外部显式调用，立即归还硬件缓冲区。
        """
        if hasattr(self, '_last_buffer') and hasattr(self, '_last_map_info'):
            self._last_buffer.unmap(self._last_map_info)
            # 显式删除引用，加速 GStreamer 内部的 unref
            del self._last_buffer
            del self._last_map_info

