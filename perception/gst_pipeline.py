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

        # 获取干净画面的拉取句柄
        self.clean_sink = self.pipeline.get_by_name("clean_sink")
        
        # 必须作为类的全局属性进行初始化，抵抗 Python 惰性 GC 的销毁
        self.meta_sink = None 
        
        self.is_running = False
        self.last_hailo_data = []

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
            f" t. ! queue name=ai_q max-size-buffers=2 leaky=downstream ! " # 在这里泄压
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"hailonet hef-path={self.hef_path} ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! " 
            f"appsink name=meta_sink emit-signals=false max-buffers=1 drop=true sync=false async=false "
        )

        # --- 5. UI 渲染业务提取分支 ---
        preview_branch = (
            f" t. ! queue name=ui_q max-size-buffers=3 leaky=downstream ! " # 为 Python 逻辑泄压
            f"appsink name=clean_sink emit-signals=false max-buffers=1 drop=false sync=false async=false "
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

    def read(self):
        """
        单管道双轨并行拉取机制。
        画面流由 Python GObject 拉取，AI 元数据流由 C++ 全局内存池提供。
        """
        if not self.is_running:
            return None, None

        clean_frame = None
        
        # === 1. 提取高清纯净原画 ===
        clean_sample = self.clean_sink.emit("try-pull-sample", 5000000) # 5 毫秒超时
        if clean_sample:
            clean_buffer = clean_sample.get_buffer()
            success, map_info = clean_buffer.map(Gst.MapFlags.READ)
            if success:
                # 执行硬拷贝，让 GStreamer 可以立刻回收该帧内存
                clean_frame = np.ndarray((self.out_h, self.out_w, 3), buffer=map_info.data, dtype=np.uint8).copy()
                clean_buffer.unmap(map_info)
            
            del map_info
            del clean_buffer
        del clean_sample 
        
        if clean_frame is None:
            return None, None

        # === 2. 高速穿透获取 C++ AI 热数据 ===
        count = hailo_bridge.get_detection_count()
        new_hailo_data = [] # 如果 count=0，直接清空上一帧缓存（消除残影 Bug）
        
        if count > 0:
            # 预申请连续内存块
            buffer_array = (DetectionBox * count)()
            hailo_bridge.get_detections(buffer_array, count)
            
            for i in range(count):
                box = buffer_array[i]
                new_hailo_data.append({
                    # 暴力切除 C 语言字符串背后的尾随空字符 (\x00)，防止 OpenCV 解析乱码
                    'label': box.label.decode('utf-8', errors='ignore').rstrip('\x00'),
                    'conf': float(box.conf),
                    'xmin': float(box.xmin), 'ymin': float(box.ymin),
                    'xmax': float(box.xmax), 'ymax': float(box.ymax)
                })
                
        # 原子替换，无需加锁
        self.last_hailo_data = new_hailo_data

        return clean_frame, getattr(self, 'last_hailo_data', [])

    def set_record_location(self, session_id):
        """动态更新录制路径。"""
        rec_sink = self.pipeline.get_by_name("rec_sink")
        if rec_sink:
            timestamp = int(time.time())
            filename = f"{session_id}_seq%05d_start{timestamp}.mp4"
            full_path = os.path.join(cfg.RECORD_SAVE_PATH, filename)
            rec_sink.set_property("location", full_path)
            logger.info(f"录制路径已成功动态更新为: {full_path}")
