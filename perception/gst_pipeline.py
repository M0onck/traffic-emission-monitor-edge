import ctypes
import gi
import cv2
import numpy as np
import os
import time
import logging
import infra.config.loader as cfg

# ==========================================
# 1. C++ 桥接层数据结构定义
# 必须与 libhailo_bridge.cpp 中的 struct 精确对齐
# ==========================================
class DetectionBox(ctypes.Structure):
    _fields_ = [
        ("xmin", ctypes.c_float),
        ("ymin", ctypes.c_float),
        ("xmax", ctypes.c_float),
        ("ymax", ctypes.c_float),
        ("conf", ctypes.c_float),
        ("label", ctypes.c_char * 64)
    ]

# ==========================================
# 2. 动态库加载与接口约束
# ==========================================
hailo_bridge = ctypes.CDLL("./bin/libhailo_bridge.so")

# 装载钩子：接管底层的 C++ 信号，绕过 Python GIL
hailo_bridge.attach_hailo_sink.argtypes = [ctypes.c_void_p]

# 卸载钩子：在停止录制时极其关键，防止底层流水线撕裂导致崩溃
hailo_bridge.detach_hailo_sink.argtypes = [ctypes.c_void_p] 

hailo_bridge.get_detection_count.restype = ctypes.c_int
hailo_bridge.get_detections.argtypes = [ctypes.POINTER(DetectionBox), ctypes.c_int]

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
    def __init__(self, config: dict, shm_array=None, frame_ready_event=None):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        Gst.init(None)

        self._shm_array = shm_array  # 直接持有共享内存引用
        self._frame_ready_event = frame_ready_event

        self.video_path = config.VIDEO_PATH
        self.hef_path = config.HEF_PATH
        self.post_so_path = config.POST_SO_PATH
        self.out_w = config.FRAME_WIDTH
        self.out_h = config.FRAME_HEIGHT
        self.use_camera = config.USE_CAMERA

        # 构建底层功能齐全（去畸变、录像、AI、预览）的并行大管道
        self.pipeline_str = self._build_pipelines()
        self.pipeline = Gst.parse_launch(self.pipeline_str)

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
                        if self._frame_ready_event:
                            self._frame_ready_event.set()
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
        
        map_bin = os.path.abspath("/home/m0onck/traffic-emission-monitor-edge/resources/dewarp_map_rgba_1456x1088.bin")

        cam_w, cam_h = 1456, 1088
        out_w, out_h = self.out_w, self.out_h
        
        # --- 1. 录像分支 ---
        record_branch = ""
        if is_camera and cfg.ENABLE_RECORD:
            os.makedirs(cfg.RECORD_SAVE_PATH, exist_ok=True)
            segment_ns = int(cfg.RECORD_SEGMENT_MIN * 60 * 1000000000)
            loc_pattern = os.path.join(cfg.RECORD_SAVE_PATH, "temp_rec_%05d.mp4")
            record_branch = (
                f" t. ! queue max-size-buffers=30 leaky=downstream ! "
                f"videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency threads=1 bitrate=2048 ! "
                f"h264parse ! splitmuxsink name=rec_sink location=\"{loc_pattern}\" max-size-time={segment_ns} async-handling=true "
            )

        if is_camera:
            source_head = f"libcamerasrc ! video/x-raw, format=NV12, width={cam_w}, height={cam_h}, framerate=30/1"
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

        # ==========================================
        # 4. AI 推理分支
        # ==========================================
        ai_branch = (
            # 1. 预处理队列 (官方建议: 大量使用QUEUE)
            f"t. ! queue name=preproc_q leaky=downstream max-size-buffers=3 ! "
            # 2. 帧率控制
            f"videorate ! video/x-raw, framerate=10/1 ! "
            # 3. 缩放与转换
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            # 4. 色彩格式转换
            f"videoconvert ! video/x-raw, format=RGB ! "
            # 5. 推理前队列
            f"queue name=inference_q leaky=downstream max-size-buffers=3 ! "
            # 6. Hailo推理
            f"hailonet hef-path={self.hef_path} "
            f"batch-size=1 "
            f"nms-score-threshold=0.3 "
            f"nms-iou-threshold=0.45 "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32 " # 使用原生量化格式
            f"! "
            # 7. 推理后队列
            f"queue name=postproc_q leaky=downstream max-size-buffers=3 ! "
            # 8. 后处理
            f"hailofilter so-path={self.post_so_path} "
            f"function-name=filter_letterbox qos=false ! "
            # 9. 最终输出 (同步属性设为false)
            f"appsink name=meta_sink emit-signals=false max-buffers=3 drop=true sync=false async=false "
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
        logger.info("正在启动 GStreamer 管道 (C++ 零拷贝桥接模式)...")
        
        # 获取底层对象并挂载至类的生命周期中，绝对防止 GC 销毁
        self.meta_sink = self.pipeline.get_by_name("meta_sink")
        
        if self.meta_sink:
            # 安全将内存地址转换为 64 位无符号正数，防止 ARM 64 位截断导致负数指针传入 C++
            sink_ptr = hash(self.meta_sink) & 0xFFFFFFFFFFFFFFFF
            # 委托 C++ 库接管管脚，彻底绕过 Python GIL
            hailo_bridge.attach_hailo_sink(ctypes.c_void_p(sink_ptr))
            
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def stop(self):
        """安全停止流水线。顺序极其严格，防止底层撕裂。"""
        if self.is_running:
            # 1. 拆除 C++ 钩子：必须在拔管之前卸载，否则仍在极速运转的 C++ 线程访问野指针会瞬间断言崩溃
            if getattr(self, 'meta_sink', None):
                sink_ptr = hash(self.meta_sink) & 0xFFFFFFFFFFFFFFFF
                hailo_bridge.detach_hailo_sink(ctypes.c_void_p(sink_ptr))
                logger.info("已安全卸载底层 C++ 钩子。")

            # 2. 发送 EOS 信号：保证所有缓冲数据落盘，mp4 文件拥有合法的 moov 尾部头
            self.pipeline.send_event(Gst.Event.new_eos())
            
            # 3. 总线等待：给予系统最多 2 秒处理完结帧
            bus = self.pipeline.get_bus()
            if bus:
                bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS)
                
            # 4. 彻底物理释放
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
            logger.info("GStreamer 流水线已安全关闭，录像封装完毕。")

    def read_metadata(self):
        """
        仅负责从 C++ 桥接层提取 AI 检测数据。
        """
        count = hailo_bridge.get_detection_count()
        new_hailo_data = []
        if count > 0:
            buffer_array = (DetectionBox * count)()
            hailo_bridge.get_detections(buffer_array, count)
            for i in range(count):
                box = buffer_array[i]
                new_hailo_data.append({
                    'label': box.label.decode('utf-8', errors='ignore').rstrip('\x00'),
                    'conf': float(box.conf),
                    'xmin': float(box.xmin), 'ymin': float(box.ymin),
                    'xmax': float(box.xmax), 'ymax': float(box.ymax)
                })
        self.last_hailo_data = new_hailo_data
        return new_hailo_data

    def read(self):
        """
        [向下兼容接口] 供主进程 UI 标定界面使用。
        返回最新缓存的画面和 AI 元数据。
        注意：在多进程引擎正式启动后，系统不应再调用此接口，而是通过共享内存读取。
        """
        # 如果还没收到第一帧，返回 None
        if self._latest_frame is None:
            return None, self.read_metadata()
            
        return self._latest_frame, self.read_metadata()

    def release_frame(self):
        """
        由外部显式调用，立即归还硬件缓冲区。
        """
        if hasattr(self, '_last_buffer') and hasattr(self, '_last_map_info'):
            self._last_buffer.unmap(self._last_map_info)
            # 显式删除引用，加速 GStreamer 内部的 unref
            del self._last_buffer
            del self._last_map_info

    def set_record_location(self, session_id):
        """动态更新录制路径。"""
        rec_sink = self.pipeline.get_by_name("rec_sink")
        if rec_sink:
            timestamp = int(time.time())
            filename = f"{session_id}_seq%05d_start{timestamp}.mp4"
            full_path = os.path.join(cfg.RECORD_SAVE_PATH, filename)
            rec_sink.set_property("location", full_path)
            logger.info(f"录制路径已成功动态更新为: {full_path}")
