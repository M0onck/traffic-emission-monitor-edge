import sys
import os
import cv2
import supervision as sv

from app.monitor_engine import TrafficMonitorEngine
from domain.vehicle.repository import VehicleRegistry
from domain.physics.vsp_calculator import VSPCalculator
from domain.physics.opmode_calculator import MovesOpModeCalculator
from domain.physics.brake_emission_model import BrakeEmissionModel
from domain.physics.tire_emission_model import TireEmissionModel
from domain.vehicle.classifier import VehicleClassifier
import infra.config.loader as cfg
from infra.store.sqlite_manager import DatabaseManager
from infra.sys.process_optimizer import SystemOptimizer
from infra.concurrency.plate_worker import PlateClassifierWorker
from perception.math.geometry import ViewTransformer
from perception.kinematics_estimator import KinematicsEstimator
from ui.renderer import Visualizer
from ui.calibration_window import CalibrationUI
from ui.console_reporter import Reporter

# --- 边缘端感知组件 ---
from perception.gst_pipeline import GstPipelineManager

def main():
    # 调用全新的主进程专属优化器 (绑定至 Core 2 并提权)
    SystemOptimizer.optimize_main_process()
    print(f"\n>>> [System] Initializing Traffic Monitor (Edge Version)...", flush=True)

    plate_worker = None  # 提前声明，方便在 finally 中清理

    try:
        calibrator = CalibrationUI(cfg.VIDEO_PATH)
        source_points, target_points = calibrator.run()
        print(f">>> [System] 标定完成", flush=True)

        print(f">>> [System] 正在组装边缘端组件...", flush=True)
        
        reporter_config = {
            "debug_mode": cfg.DEBUG_MODE,
            "fps": cfg.FPS,
            "min_survival_frames": cfg.MIN_SURVIVAL_FRAMES
        }

        classifier_config = {
            "car": cfg.YOLO_CLASS_CAR,
            "bus": cfg.YOLO_CLASS_BUS,
            "truck": cfg.YOLO_CLASS_TRUCK
        }

        vsp_config = {
            "vsp_coefficients": cfg.VSP_COEFFS,
            "road_grade_percent": cfg.ROAD_GRADE_PERCENT
        }
        
        # 共享 OpMode 计算器
        opmode_calculator = MovesOpModeCalculator(config=cfg._e)

        # 刹车与轮胎模型配置
        brake_emission_config = {
            "braking_decel_threshold": cfg.BRAKING_DECEL_THRESHOLD,
            "idling_speed_threshold": cfg.IDLING_SPEED_THRESHOLD,
            "low_speed_threshold": cfg.LOW_SPEED_THRESHOLD,
            "mass_factor_ev": cfg.MASS_FACTOR_EV,
            "brake_wear_coefficients": cfg.BRAKE_WEAR_COEFFICIENTS 
        }

        tire_emission_config = {
            "tire_wear_coefficients": cfg.TIRE_WEAR_COEFFICIENTS,
            "emission_params": cfg._e
        }
        
        kinematics_config = {
            "fps": cfg.FPS,
            "kinematics": {
                "speed_window": cfg.SPEED_WINDOW,
                "accel_window": cfg.ACCEL_WINDOW,
                "border_margin": cfg.BORDER_MARGIN,
                "min_tracking_frames": cfg.MIN_TRACKING_FRAMES,
                "max_physical_accel": cfg.MAX_PHYSICAL_ACCEL,
                "poly_order": getattr(cfg, "KINEMATICS_POLY_ORDER", 3)
            }
        }

        # --- 初始化 GStreamer 管道 ---
        gst_config = {
            "video_path": cfg.VIDEO_PATH,
            "hef_path": getattr(cfg, "HEF_PATH", "resources/yolov8m.hef"),
            "post_so_path": getattr(cfg, "POST_SO_PATH", "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so")
        }
        camera_manager = GstPipelineManager(gst_config)

        # --- 基础组件字典 ---
        components = {
            'camera': camera_manager,
            'smoother': sv.DetectionsSmoother(length=3),
            'transformer': ViewTransformer(source_points, target_points),
            'visualizer': Visualizer(
                calibration_points=source_points,
                trace_length=cfg.FPS,
                opmode_calculator=opmode_calculator
            ),
            'registry': VehicleRegistry(
                fps=cfg.FPS,
                min_survival_frames=cfg.MIN_SURVIVAL_FRAMES,
                exit_threshold=cfg.EXIT_THRESHOLD,
                min_valid_pts=cfg.MIN_VALID_POINTS,
                min_moving_dist=cfg.MIN_MOVING_DIST
            ),
            'db': DatabaseManager(cfg.DB_PATH, cfg.FPS),
            'classifier': VehicleClassifier(cfg.TYPE_MAP, classifier_config)
        }

        if cfg.DEBUG_MODE:
            components['reporter'] = Reporter(reporter_config, opmode_calculator)

        if cfg.ENABLE_MOTION:
            components['kinematics'] = KinematicsEstimator(kinematics_config)
        
        if cfg.ENABLE_EMISSION and cfg.ENABLE_MOTION:
            components['vsp_calculator'] = VSPCalculator(vsp_config)
            components['brake_model'] = BrakeEmissionModel(brake_emission_config)
            components['tire_model'] = TireEmissionModel(tire_emission_config, opmode_calculator)
            
        # 按需初始化多进程 Worker，彻底删除旧版同步模型
        if cfg.ENABLE_OCR:
            y5fu_path = getattr(cfg, "Y5FU_PATH", "perception/plate_classifier/models/y5fu_320x_sim.onnx")
            litemodel_path = getattr(cfg, "LITEMODEL_PATH", "perception/plate_classifier/models/litemodel_cls_96x_r1.onnx")
            
            plate_worker = PlateClassifierWorker(
                detector_path=y5fu_path,
                classifier_path=litemodel_path,
                max_queue_size=10
            )
            plate_worker.start()
            # 将 worker 注入组件字典，供 monitor_engine 调用
            components['plate_worker'] = plate_worker

        engine = TrafficMonitorEngine(cfg, components)
        engine.run()

    except KeyboardInterrupt:
        if 'engine' in locals(): engine.cleanup(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'engine' in locals(): engine.cleanup(0)
    finally:
        # 无论程序是正常结束还是报错崩溃，确保杀掉独立的 OCR 子进程
        if plate_worker is not None:
            print("\n>>> [System] 正在安全关闭异步分类子进程...", flush=True)
            plate_worker.stop()

if __name__ == "__main__":
    main()