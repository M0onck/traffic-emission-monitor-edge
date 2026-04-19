import gi
import cv2
import numpy as np
import os
import logging
import infra.config.loader as cfg

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

        # 仅构建一条单管道
        self.pipeline_str = self._build_pipelines()
        self.pipeline = Gst.parse_launch(self.pipeline_str)

        # 获取数据终点
        self.clean_sink = self.pipeline.get_by_name("clean_sink")
        self.meta_sink = self.pipeline.get_by_name("meta_sink")
        self.is_running = False

        # 缓存上一帧的 AI 数据
        self.last_hailo_data = []

    def _build_pipelines(self):
        is_camera = getattr(self, 'use_camera', False) or self.video_path.startswith("libcamerasrc") or self.video_path.startswith("v4l2src")
        
        # 镜头原生分辨率标定文件（binary）
        map_bin = os.path.abspath("/home/m0onck/traffic-emission-monitor-edge/resources/dewarp_map_rgba_1456x1088.bin")

        # 定义底层物理分辨率和目标输出分辨率
        cam_w, cam_h = 1456, 1088
        out_w, out_h = self.out_w, self.out_h
        
        # 1. 录制分支
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
            # dewarpfilter 是自定义的 GPU 去畸变组件
            f"dewarpfilter map-file-path={map_bin} map-width={cam_w} map-height={cam_h} ! " # GPU 处理段，此时流是 1456x1088
            f"glcolorconvert ! gldownload ! videoconvert ! "
            f"videocrop top={crop_top} bottom={crop_bottom} left={crop_left} right={crop_right} ! " # 裁剪段，切出中心画幅
            f"video/x-raw, width={out_w}, height={out_h}, format=BGR ! tee name=t" # 锁定格式进入分支
        )

        # 3. AI 推理分支
        ai_branch = (
            f" t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"videorate ! video/x-raw, framerate=10/1 ! "
            f"videoscale ! video/x-raw, width=640, height=640 ! "
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"hailonet hef-path={self.hef_path} ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! "
            f"queue max-size-buffers=5 leaky=downstream ! "
            f"appsink name=meta_sink emit-signals=false max-buffers=1 drop=true sync=false"
        )

        # 4. 画面预览分支
        preview_branch = (
            f" t. ! queue min-threshold-time=100000000 max-size-buffers=5 leaky=downstream ! "
            f"appsink name=clean_sink emit-signals=false max-buffers=1 drop=true sync=false"
        )

        final_pipeline_str = f"{source_section}{ai_branch}{preview_branch}{record_branch}"
        
        logger.info(f"构建 GPU 加速管道: {final_pipeline_str}")
        return final_pipeline_str

    def start(self):
        # 现在的架构只有一条主管道，直接启动即可
        logger.info("正在启动 GPU 加速 GStreamer 管道...")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def stop(self):
        if self.is_running:
            # 1. 向整条单管道发送 EOS 信号
            # 这样信号会顺着管道流向 rec_sink (splitmuxsink)，确保录像安全封口
            self.pipeline.send_event(Gst.Event.new_eos())
            
            # 2. 等待 EOS 消息到达总线（最多等待 2 秒）
            bus = self.pipeline.get_bus()
            bus.timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS)
            
            # 3. 将整条管道设为 NULL 状态以释放硬件资源（Camera, GPU, Hailo, NPU）
            self.pipeline.set_state(Gst.State.NULL)
            
            self.is_running = False
            logger.info("GStreamer 硬件流水线已安全关闭。")

    def read(self):
        """单管道模式：直接从底层并行提取已去畸变的干净画面和 AI 结果"""
        if not self.is_running:
            return None, None

        clean_frame = None
        
        # ==========================================
        # 1. 提取画面分支 (必须极其严苛地管理 PyGObject 引用)
        # ==========================================
        clean_sample = self.clean_sink.emit("try-pull-sample", 5000000) # 5毫秒超时
        if clean_sample:
            clean_buffer = clean_sample.get_buffer()
            success, map_info = clean_buffer.map(Gst.MapFlags.READ)
            if success:
                # 极速拷贝底层内存到 Python 空间 (完全切断物理联系)
                clean_frame = np.ndarray((self.out_h, self.out_w, 3), buffer=map_info.data, dtype=np.uint8).copy()
                clean_buffer.unmap(map_info)
            
            # 立刻强杀 Python 到 C 的代理对象，强迫 GStreamer 引用计数 -1
            del map_info
            del clean_buffer
        
        # 无论成功与否，必须释放 Sample 归还给上游池！
        del clean_sample 
        
        # 如果连基础画面都没拿到，直接返回，跳过 AI 解析
        if clean_frame is None:
            return None, None

        # ==========================================
        # 2. 提取 AI 元数据分支
        # ==========================================
        meta_sample = self.meta_sink.emit("try-pull-sample", 5000000) 
        
        if meta_sample:
            new_hailo_data = []
            meta_buffer = meta_sample.get_buffer()
            try:
                import hailo
                roi = hailo.get_roi_from_buffer(meta_buffer)
                hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
                
                for det in hailo_detections:
                    bbox = det.get_bbox()
                    new_hailo_data.append({
                        'label': det.get_label(),
                        'conf': det.get_confidence(),
                        'xmin': bbox.xmin(), 'ymin': bbox.ymin(),
                        'xmax': bbox.xmax(), 'ymax': bbox.ymax()
                    })
                    # 循环内释放临时底层对象的引用
                    del bbox
                    del det
                
                # 析构 Hailo NPU 结构体引用
                del hailo_detections
                del roi
            except Exception:
                pass 
            
            self.last_hailo_data = new_hailo_data

            # 释放 Hailo 的 Meta GstBuffer
            del meta_buffer
            
        # 无论成功与否，必须释放 Meta Sample 归还给 HailoRT 插件
        del meta_sample

        return clean_frame, self.last_hailo_data

    def set_record_location(self, session_id):
        rec_sink = self.pipeline.get_by_name("rec_sink")
        if rec_sink:
            import time
            timestamp = int(time.time())
            filename = f"{session_id}_seq%05d_start{timestamp}.mp4"
            full_path = os.path.join(cfg.RECORD_SAVE_PATH, filename)
            rec_sink.set_property("location", full_path)
            logger.info(f"录制路径已更新为: {full_path}")
