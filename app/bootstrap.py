# app/bootstrap.py
import numpy as np
import multiprocessing as mp
from infra.store.sqlite_manager import DatabaseManager
from infra.concurrency.async_recognizer import AsyncPlateRecognizer
from domain.vehicle.repository import VehicleRegistry
from domain.vehicle.classifier import VehicleClassifier
from perception.gst_pipeline import GstPipelineManager
from perception.sensor.thermal_camera import ThermalCamera
from ui.renderer import Visualizer

# 引入对齐守护进程
from app.alignment_daemon import AlignmentDaemon

class AppBootstrap:
    """
    [应用引导程序] 负责系统组件的实例化、依赖注入与多进程 IPC 通道建立。
    """
    
    @staticmethod
    def _run_alignment_worker(config, sync_queue, stop_event):
        """对齐守护进程的入口隔离函数"""
        daemon = AlignmentDaemon(config, sync_queue, stop_event)
        daemon.run()

    @staticmethod
    def setup_components(config):
        print(">>> [Bootstrap] 正在组装系统组件与 IPC 通道...")
        
        # --- 核心新增：全局通信管道与控制信号 ---
        ctx = mp.get_context('spawn')
        sync_queue = ctx.Queue(maxsize=10) # 限制容量防内存溢出
        stop_event = ctx.Event()

        # 1. 基础设施层
        db = DatabaseManager(db_path=config.DB_PATH, fps=config.FPS)
        force_delay = config.ALIGNMENT_DELAY_SEC if config.RUN_MODE == 'inference' else float('inf')

        # 2. 领域层
        registry = VehicleRegistry(
            target_fps=config.FPS,
            min_survival_sec=config.MIN_SURVIVAL_SEC,
            exit_timeout_sec=config.EXIT_TIMEOUT_SEC,
            min_valid_pts=config.MIN_VALID_POINTS,
            min_moving_dist=config.MIN_MOVING_DIST,
            force_delay_sec=force_delay
        )
        classifier = VehicleClassifier(yolo_classes={
            'car': config.YOLO_CLASS_CAR, 'bus': config.YOLO_CLASS_BUS, 'truck': config.YOLO_CLASS_TRUCK
        })

        # 3. 异步处理与传感器
        plate_worker = AsyncPlateRecognizer() if getattr(config, 'ENABLE_OCR', False) else None
        lib_path = getattr(config, 'THERMAL_LIB_PATH', 'bin/libmlx90640.so')
        thermal_cam = ThermalCamera(lib_path)

        # 4. 渲染层
        target_pts_raw = getattr(config, 'TARGET_POINTS', [[0,0], [1,0], [1,1], [0,1]])
        target_points = np.array(target_pts_raw, dtype=np.float32)
        norm_source_points = np.array(config.SOURCE_POINTS, dtype=np.float32) if getattr(config, 'SOURCE_POINTS', None) else None
        visualizer = Visualizer(calibration_points=target_points, target_fps=config.FPS)

        # --- 核心新增：按需启动后台对齐进程 ---
        alignment_proc = None
        if config.RUN_MODE == 'inference':
            alignment_proc = ctx.Process(
                target=AppBootstrap._run_alignment_worker,
                args=(config, sync_queue, stop_event),
                daemon=True
            )
            alignment_proc.start() # 在此处启动，由于是阻塞队列，它会安静地等待
            print(">>> [Bootstrap] 延迟对齐后台进程已挂载。")

        # 5. 封装最终字典
        components = {
            'db': db,
            'registry': registry,
            'classifier': classifier,
            'plate_worker': plate_worker,
            'visualizer': visualizer,
            'target_points': target_points,
            'norm_source_points': norm_source_points,
            'thermal_cam': thermal_cam,
            
            # 注入 IPC 通信与控制对象
            'sync_queue': sync_queue,
            'stop_event': stop_event,
            'alignment_proc': alignment_proc
        }

        print(">>> [Bootstrap] 组件组装完成。")
        return components
