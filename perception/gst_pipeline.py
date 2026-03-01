import gi
import cv2
import numpy as np
import time
import os

# 初始化 GStreamer 环境
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class GstPipelineManager:
    """
    [感知层] GStreamer 管道管理器
    功能：构建并管理底层硬件加速流水线。
    数据流向：本地 1080p 视频 -> 硬件解码 -> Hailo检测(含关键点) -> 目标追踪 -> Appsink(Python)
    """
    def __init__(self, config: dict):
        # 强制使用 X11 后端以防潜在的显示异常
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        Gst.init(None)

        # 从配置中读取路径 (开发阶段可先硬编码测试)
        self.video_path = config.get("video_path", "resources/test_traffic.mp4")
        self.hef_path = config.get("hef_path", "cpp_postprocess/y5fu_320x_sim.hef")
        self.post_so_path = config.get("post_so_path", "cpp_postprocess/build/liby5fu_post.so")
        
        # 管道构建逻辑
        self.pipeline_string = self._build_pipeline()
        self.pipeline = Gst.parse_launch(self.pipeline_string)
        self.appsink = self.pipeline.get_by_name("mysink")
        
        # 状态标志
        self.is_running = False

    def _build_pipeline(self) -> str:
        """
        构建针对 1080p 视频文件的混合架构管道。
        注意：没有使用 tee 分流，主链路保持 1080p 传递给 appsink，以保证车牌透视变换的清晰度。
        """
        # filesrc 与 decodebin 用于读取和解码本地视频文件
        pipeline = (
            f"filesrc location={self.video_path} ! decodebin ! "
            
            # 统一转换为 RGB 格式并确保输出 1080p 分辨率
            f"videoconvert ! video/x-raw, format=RGB ! "
            f"videoscale ! video/x-raw, width=1920, height=1080 ! "
            
            # 第一阶段：硬件检测 (hailonet 内部会自动缩放至 320x320，但不改变外层 Buffer 尺寸)
            f"hailonet hef-path={self.hef_path} ! "
            f"hailofilter so-path={self.post_so_path} qos=false ! "
            
            # 第二阶段：目标追踪 (直接在 GStreamer 管道中完成，替代原有的 kalman_filter)
            f"hailotracker name=hailo_tracker keep-tracked-frames=3 class-id=-1 ! "
            
            # 格式转换：转为 BGR 供 Python 层的 OpenCV 直接使用
            f"videoconvert ! video/x-raw, format=RGBA ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            
            # 终点：丢弃旧帧，保证 Python 层拿到的是最新帧
            f"appsink name=mysink emit-signals=false max-buffers=1 drop=true"
        )
        return pipeline

    def start(self):
        """启动管道"""
        print(f"🚀 正在启动 GStreamer 管道: 读取 {self.video_path}")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True

    def stop(self):
        """安全停止管道"""
        if self.is_running:
            print("🛑 正在停止管道...")
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False

    def read(self):
        """
        供 monitor_engine 调用的拉取接口
        返回: (numpy.ndarray 图像帧, Gst.Buffer 元数据缓冲)
        """
        if not self.is_running:
            return None, None

        # 阻塞式拉取最新处理好的样本
        sample = self.appsink.emit("pull-sample")
        if not sample:
            return None, None

        buffer = sample.get_buffer()
        
        try:
            # 提取 1080p 的 BGR 图像数据
            buffer_map = buffer.map(Gst.MapFlags.READ)
            # 根据管道末端的设定，这里是 1080p BGR
            frame = np.ndarray(
                (1080, 1920, 3), 
                buffer=buffer_map[1].data, 
                dtype=np.uint8
            ).copy()  # 深拷贝以防止 GStreamer 内存重写
            buffer.unmap(buffer_map[1])
            
            # 将 frame 和携带 Metadata 的 buffer 一并返回给引擎层
            return frame, buffer
            
        except Exception as e:
            print(f"Buffer 解析错误: {e}")
            return None, None
