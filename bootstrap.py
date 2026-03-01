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
from perception.math.geometry import ViewTransformer
from perception.kinematics_estimator import KinematicsEstimator
from ui.renderer import Visualizer
from ui.calibration_window import CalibrationUI
from ui.console_reporter import Reporter

# --- 新增/保留的边缘端感知组件 ---
# 1. 硬件加速管道管理器
from perception.gst_pipeline import GstPipelineManager
# 2. 恢复保留的车牌检测与分类模型 (运行在 Python 端 CPU)
from perception.plate_classifier.core.multitask_detect import MultiTaskDetectorORT
from perception.plate_classifier.core.classification import ClassificationORT
from perception.plate_classifier.pipeline import EdgePlateClassifierPipeline

def main():
    SystemOptimizer.set_cpu_affinity("main")
    print(f"\n>>> [System] Initializing Traffic Monitor (Edge Version)...", flush=True)

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

        # --- 初始化 GStreamer 管道 (使用官方 YOLOv8 抓车辆) ---
        gst_config = {
            "video_path": cfg.VIDEO_PATH,
            "hef_path": getattr(cfg, "HEF_PATH", "resources/yolov8m.hef"),
            "post_so_path": getattr(cfg, "POST_SO_PATH", "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so")
        }
        camera_manager = GstPipelineManager(gst_config)

        # --- 组件字典大换血 ---
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
            
        if cfg.ENABLE_OCR:
            # --- 恢复双模型架构的轻量级调用 (Python端) ---
            y5fu_path = getattr(cfg, "Y5FU_PATH", "perception/plate_classifier/models/y5fu_320x_sim.onnx")
            litemodel_path = getattr(cfg, "LITEMODEL_PATH", "perception/plate_classifier/models/litemodel_cls_96x_r1.onnx")
            
            detector = MultiTaskDetectorORT(y5fu_path)
            classifier = ClassificationORT(litemodel_path)
            components['plate_classifier'] = EdgePlateClassifierPipeline(detector, classifier)

        engine = TrafficMonitorEngine(cfg, components)
        engine.run()

    except KeyboardInterrupt:
        if 'engine' in locals(): engine.cleanup(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'engine' in locals(): engine.cleanup(0)

if __name__ == "__main__":
    main()
