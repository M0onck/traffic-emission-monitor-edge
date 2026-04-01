# app/bootstrap.py
import numpy as np
from infra.store.sqlite_manager import DatabaseManager
from infra.concurrency.async_recognizer import AsyncPlateRecognizer
from domain.vehicle.repository import VehicleRegistry
from domain.vehicle.classifier import VehicleClassifier
from perception.gst_pipeline import GstPipelineManager
from perception.sensor.thermal_camera import ThermalCamera
from ui.renderer import Visualizer

class AppBootstrap:
    """
    [应用引导程序] 负责系统组件的实例化与依赖注入装配。
    """
    @staticmethod
    def setup_components(config):
        print(">>> [Bootstrap] 正在组装系统组件...")

        # 1. 基础设施层：数据库 (补充缺失的 fps 参数)
        db = DatabaseManager(db_path=config.DB_PATH, fps=config.FPS)

        # 2. 领域层：注册表 (注入配置文件中的核心业务阈值)
        registry = VehicleRegistry(
            fps=config.FPS,
            min_survival_frames=config.MIN_SURVIVAL_FRAMES,
            exit_threshold=config.EXIT_THRESHOLD,
            min_valid_pts=config.MIN_VALID_POINTS,
            min_moving_dist=config.MIN_MOVING_DIST
        )
        
        # 组装分类器所需的 yolo_classes 字典
        yolo_classes_dict = {
            'car': config.YOLO_CLASS_CAR,
            'bus': config.YOLO_CLASS_BUS,
            'truck': config.YOLO_CLASS_TRUCK
        }
        classifier = VehicleClassifier(type_map=config.TYPE_MAP, yolo_classes=yolo_classes_dict)

        # 3. 感知层：GStreamer 视频流硬件管道 (组装成期待的 dict)
        camera_cfg = {
            "video_path": config.VIDEO_PATH,
            "hef_path": config.HEF_PATH,
            "post_so_path": config.POST_SO_PATH
        }
        camera = GstPipelineManager(camera_cfg)

        # 4. 异步处理：OCR 车牌识别工人 (传入具体的模型路径)
        plate_worker = None
        if config.ENABLE_OCR:
            print(">>> [Bootstrap] 启动异步 OCR 识别引擎...")
            plate_worker = AsyncPlateRecognizer(
                y5fu_onnx_path=config.Y5FU_PATH,
                litemodel_onnx_path=config.LITEMODEL_PATH
            )
            # 注意：此处移除了 plate_worker.start()，因为在 async_recognizer.py 中
            # __init__ 已经调用了 self.worker_process.start()，避免重复启动进程报错

        # 5. 空间坐标标定数据加载 (必须在 Visualizer 初始化之前解析)
        # 注意：此处增加安全获取，防止 TARGET_POINTS 在某些精简配置下不存在
        target_pts_raw = getattr(config, 'TARGET_POINTS', [[0,0], [1,0], [1,1], [0,1]])
        target_points = np.array(target_pts_raw, dtype=np.float32)
        
        norm_source_points = None
        if hasattr(config, 'SOURCE_POINTS') and config.SOURCE_POINTS:
            norm_source_points = np.array(config.SOURCE_POINTS, dtype=np.float32)

        # 6. 表示层：可视化渲染器 (传入解析好的 numpy 数组)
        visualizer = Visualizer(calibration_points=target_points)

        # 7. 初始化热成像模块
        lib_path = getattr(config, 'THERMAL_LIB_PATH', './libmlx90640.so')
        thermal_cam = ThermalCamera(lib_path)

        # 8. 封装组件字典
        components = {
            'db': db,
            'registry': registry,
            'camera': camera,
            'classifier': classifier,
            'plate_worker': plate_worker,
            'visualizer': visualizer,
            'target_points': target_points,
            'norm_source_points': norm_source_points,
            'thermal_cam': thermal_cam
        }

        print(">>> [Bootstrap] 组件组装完成。")
        return components
