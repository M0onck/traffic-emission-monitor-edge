import ctypes

class DetectionBox(ctypes.Structure):
    _fields_ = [
        ("xmin", ctypes.c_float),
        ("ymin", ctypes.c_float),
        ("xmax", ctypes.c_float),
        ("ymax", ctypes.c_float),
        ("conf", ctypes.c_float),
        ("label", ctypes.c_char * 64)
    ]

# 全局加载动态库
hailo_bridge = ctypes.CDLL("./bin/libhailo_bridge.so")
hailo_bridge.attach_hailo_sink.argtypes = [ctypes.c_void_p]
hailo_bridge.get_detection_count.restype = ctypes.c_int
hailo_bridge.get_detections.argtypes = [ctypes.POINTER(DetectionBox), ctypes.c_int]

import gi
import cv2
import numpy as np
import os
import time
import logging
import infra.config.loader as cfg

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# ==========================================
# [终极安全保障 1] 必须在全局域提前加载 NPU 绑定
# 绝不能在 GStreamer 异步回调的运行期去 import，防止 GIL 引发段错误！
# ==========================================
try:
    import hailo
except ImportError:
    logging.warning("Hailo module not found. AI features will fail.")

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

        # 构建底层功能齐全（去畸变、录像、AI、预览）的并行大管道
        self.pipeline_str = self._build_pipelines()
        self.pipeline = Gst.parse_launch(self.pipeline_str)

        # 获取数据终点句柄
        self.clean_sink = self.pipeline.get_by_name("clean_sink")
        self.meta_sink = self.pipeline.get_by_name("meta_sink")
        self.is_running = False

        # 缓存上一帧的 AI 数据，供 Python 业务线程非阻塞读取
        self.last_hailo_data = []

        # ==========================================
        # [终极安全保障 2] 接管原生异步信号
        # ==========================================
        if self.meta_sink:
            self.meta_sink.connect("new-sample", self._on_new_meta_sample)

    def _build_pipelines(self):
        is_camera = getattr(self, 'use_camera', False) or self.video_path.startswith("libcamerasrc") or self.video_path.startswith("v4l2src")
        
        # 镜头原生分辨率标定文件（binary）
        map_bin = os.path.abspath("/home/m0onck/traffic-emission-monitor-edge/resources/dewarp_map_rgba_1456x1088.bin")

        # 定义底层物理分辨率和目标输出分辨率
        cam_w, cam_h = 1456, 1088
        out_w, out_h = self.out_w, self.out_h
        
        # 1. 录制分支 (保留了分段录像和 x264 硬件/软加速配置)
        record_branch = ""
        if is_camera and cfg.ENABLE_RECORD:
            os.makedirs(cfg.RECORD_SAVE_PATH, exist_ok=True)
            segment_ns = int(cfg.RECORD_SEGMENT_MIN * 60 * 1000000000)
            loc_pattern = os.path.join(cfg.RECORD_SAVE_PATH, "temp_rec_%05d.mp4")
            # 增加 async-handling=true 防止写盘时拖慢主管道
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

        # 画幅裁剪参数计算
        crop_left = (cam_w - out_w) // 2
        crop_right = cam_w - out_w - crop_left
        crop_top = (cam_h - out_h) // 2
        crop_bottom = cam_h - out_h - crop_top

        # 2. 源流与 GPU 去畸变主干，动态接入 source_head
        source_section = (
            f"{source_head} ! "
            f"glupload ! glcolorconvert ! "
            f"dewarpfilter map-file-path={map_bin} map-width={cam_w} map-height={cam_h} ! " 
            f"glcolorconvert ! gldownload ! videoconvert ! "
            f"videocrop top={crop_top} bottom={crop_bottom} left={crop_left} right={crop_right} ! " 
            f"video/x-raw, width={out_w}, height={out_h}, format=BGR ! tee name=t " 
        )

        # 3. AI 推理分支
        # [核心修复] drop=false 保护 NPU 内存池不被白嫖；emit-signals=true 触发信号；async=false 破除预卷死锁
        ai_branch = (
            f" t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"videorate ! video/x-raw, framerate=10/1 ! "
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"hailonet hef-path={self.hef_path} ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! "
            f"queue max-size-buffers=3 ! "
            f"appsink name=meta_sink emit-signals=true max-buffers=1 drop=false sync=false async=false "
        )

        # 4. 画面预览/业务提取分支
        # [核心修复] 加入 async=false，如果漏掉这个，UI 线程仍然会因为等不到纯净帧而短暂卡死
        preview_branch = (
            f" t. ! queue min-threshold-time=100000000 max-size-buffers=5 leaky=downstream ! "
            f"appsink name=clean_sink emit-signals=false max-buffers=2 drop=true sync=false async=false "
        )

        final_pipeline_str = f"{source_section}{ai_branch}{preview_branch}{record_branch}"
        
        logger.info(f"构建 GPU 加速且安全的重构管道: {final_pipeline_str}")
        return final_pipeline_str

    def start(self):
        logger.info("正在启动 GStreamer 管道 (无锁模式)...")
        # 由于两路 appsink 都配置了 async=false，这一行会极速返回，彻底解放 PyQt GUI 线程
        meta_sink = self.pipeline.get_by_name("meta_sink")
        if meta_sink:
            # PyGObject 绝技：使用 hash() 直接获取底层 GstElement 的纯 C 指针
            sink_ptr = hash(meta_sink)
            hailo_bridge.attach_hailo_sink(ctypes.c_void_p(sink_ptr))
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def stop(self):
        if self.is_running:
            # 1. 向整条单管道发送 EOS 信号，确保 mp4 视频拥有合法的 mp4 moov 原子结束头，否则录像文件会损坏
            self.pipeline.send_event(Gst.Event.new_eos())
            
            # 2. 等待 EOS 消息到达总线（最多等待 2 秒）
            bus = self.pipeline.get_bus()
            bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS)
            
            # 3. 将整条管道设为 NULL 状态以释放硬件资源（Camera, GPU, Hailo, NPU）
            self.pipeline.set_state(Gst.State.NULL)
            
            self.is_running = False
            logger.info("GStreamer 硬件流水线已安全关闭，录像已封装。")

    def _on_new_meta_sample(self, sink):
        """
        [C++ 线程异步回调] NPU 一有推理结果就主动推送，绝不让 Python 主线程干等。
        """
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        try:
            buffer = sample.get_buffer()
            roi = hailo.get_roi_from_buffer(buffer)
            hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
            
            new_hailo_data = []
            for det in hailo_detections:
                bbox = det.get_bbox()
                new_hailo_data.append({
                    'label': det.get_label(),
                    'conf': det.get_confidence(),
                    'xmin': bbox.xmin(), 'ymin': bbox.ymin(),
                    'xmax': bbox.xmax(), 'ymax': bbox.ymax()
                })
            
            # 原子的引用替换操作，完美避免 GIL 锁竞争
            self.last_hailo_data = new_hailo_data
            
        except Exception as e:
            logger.error(f"[Hailo Signal Error] {e}")

        # 放行，C++ 底层自动且安全地回收 RequestWrap 内存！
        return Gst.FlowReturn.OK

    def read(self):
        """
        纯净数据拉取：只负责极速拉取干净画面。
        AI 数据已被底层的 C++ 信号伺服静默更新。
        """
        if not self.is_running:
            return None, None

        clean_frame = None
        
        # 画面分支 (严苛管理 PyGObject)
        clean_sample = self.clean_sink.emit("try-pull-sample", 5000000) # 5ms 超时
        if clean_sample:
            clean_buffer = clean_sample.get_buffer()
            success, map_info = clean_buffer.map(Gst.MapFlags.READ)
            if success:
                # 拷贝到底层断开物理连接
                clean_frame = np.ndarray((self.out_h, self.out_w, 3), buffer=map_info.data, dtype=np.uint8).copy()
                clean_buffer.unmap(map_info)
            
            del map_info
            del clean_buffer
            
        del clean_sample 
        
        if clean_frame is None:
            return None, None
        
        # 瞬间穿透内存，提取最新的 AI 数据
        count = hailo_bridge.get_detection_count()
        if count > 0:
            # 在内存中预分配 ctypes 数组结构
            buffer_array = (DetectionBox * count)()
            hailo_bridge.get_detections(buffer_array, count)
            
            new_hailo_data = []
            for i in range(count):
                box = buffer_array[i]
                new_hailo_data.append({
                    'label': box.label.decode('utf-8'),
                    'conf': float(box.conf),
                    'xmin': float(box.xmin), 'ymin': float(box.ymin),
                    'xmax': float(box.xmax), 'ymax': float(box.ymax)
                })
            self.last_hailo_data = new_hailo_data

        # 组合画面与 AI 热数据返回给 monitor_engine.py
        return clean_frame, getattr(self, 'last_hailo_data', [])

    def set_record_location(self, session_id):
        # 恢复录像路径动态设定的功能
        rec_sink = self.pipeline.get_by_name("rec_sink")
        if rec_sink:
            timestamp = int(time.time())
            filename = f"{session_id}_seq%05d_start{timestamp}.mp4"
            full_path = os.path.join(cfg.RECORD_SAVE_PATH, filename)
            rec_sink.set_property("location", full_path)
            logger.info(f"录制路径已成功更新为: {full_path}")
