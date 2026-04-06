import json
import os
import sys

"""
[基础层] 配置文件读取器
功能：负责读取外部 config.json 配置文件，并将其解析为 Python 原生数据类型。
"""

CONFIG_FILE = "config.json"

try:
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"配置文件 '{CONFIG_FILE}' 未找到")
        
    with open(CONFIG_FILE, "r", encoding='utf-8') as f:
        _cfg = json.load(f)
        
except Exception as e:
    print(f"[Settings] Error: 配置文件加载失败 - {e}")
    sys.exit(1)

# --- 1. 基础参数 ---
_sys = _cfg["system"]
VIDEO_PATH = _sys["video_path"]
TARGET_VIDEO_PATH = _sys["target_video_path"]
DB_PATH = _sys["db_path"]
FPS = _sys["fps"]
DEBUG_MODE = _sys["debug_mode"]
USE_CAMERA = _sys.get("use_camera", False)

_d = _cfg["display"]
FRAME_WIDTH = _d.get("frame_width", 1280)
FRAME_HEIGHT = _d.get("frame_height", 720)

_sw = _cfg["switches"]
ENABLE_MOTION = _sw["enable_motion"]
ENABLE_OCR = _sw["enable_ocr"]

_k = _cfg["kinematics"]
SPEED_WINDOW_SEC = _k.get("speed_window_sec", 2.2)
ACCEL_WINDOW_SEC = _k.get("accel_window_sec", 3.0)
BORDER_MARGIN = _k["border_margin"]
MIN_TRACKING_SEC = _k.get("min_tracking_sec", 0.3)
MAX_PHYSICAL_ACCEL = _k["max_physical_accel"]
MIN_SURVIVAL_SEC = _k.get("min_survival_sec", 1.0)
EXIT_TIMEOUT_SEC = _k.get("exit_timeout_sec", 1.0)
KINEMATICS_POLY_ORDER = _k.get("poly_order", 3)

_o = _cfg["ocr_params"]
OCR_RETRY_COOLDOWN = _o["retry_cooldown"]
OCR_INTERVAL = _o["run_interval"]
OCR_CONF_THRESHOLD = _o["confidence_threshold"]

_p = _cfg.get("physics_params", {})
ROAD_GRADE_PERCENT = _p.get("road_grade_percent", 0.0)

# 质量控制参数
_qc = _cfg.get("quality_control", {})
MIN_VALID_POINTS = _qc.get("min_valid_trajectory_points", 15)
QC_MIN_SURVIVAL_SEC = _qc.get("min_survival_sec", 1.0)
MIN_MOVING_DIST = _qc.get("min_moving_distance_m", 2.0)
BLUR_THRESHOLD = _qc.get("blur_threshold", 100.0)

# --- 2. 核心常量 ---
_y = _cfg["yolo_classes"]
YOLO_CLASS_CAR = _y["yolo_class_car"]
YOLO_CLASS_BUS = _y["yolo_class_bus"]
YOLO_CLASS_TRUCK = _y["yolo_class_truck"]
YOLO_INTEREST_CLASSES = [YOLO_CLASS_CAR, YOLO_CLASS_BUS, YOLO_CLASS_TRUCK]

VEHICLE_SEMANTIC_MAP = {
    "car": YOLO_CLASS_CAR,
    "bus": YOLO_CLASS_BUS,
    "truck": YOLO_CLASS_TRUCK
}

# --- 3. 边缘端特有模型路径 ---
_edge = _cfg.get("edge_models", {})
HEF_PATH = _edge.get("hef_path", "resources/yolov8m.hef")
POST_SO_PATH = _edge.get("post_so_path", "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so")
Y5FU_PATH = _edge.get("y5fu_path", "perception/plate_classifier/models/y5fu_320x_sim.onnx")
LITEMODEL_PATH = _edge.get("litemodel_path", "perception/plate_classifier/models/litemodel_cls_96x_r1.onnx")

# --- 4. 延迟对齐与时间窗参数 (time_windows) ---
_tw = _cfg.get("time_windows", {})
ALIGNMENT_DELAY_SEC = _tw.get("alignment_delay_sec", 60.0)
INTEGRATION_WINDOW_SEC = _tw.get("integration_window_sec", 30.0)
BASELINE_WINDOW_MINUTE = _tw.get("baseline_window_minute", 10.0)
DB_ALIGN_FREQUENCY_HZ = _tw.get("db_align_frequency_hz", 1.0)

# --- 5. 物理与环境先验参数 (physics_priors) ---
_pp = _cfg.get("physics_priors", {})
WEATHER_STATION_X_POS = _pp.get("weather_station_x_pos", 0.0)
ROAD_DIRECTION_ANGLE = _pp.get("road_direction_angle", 0.0)
NEV_MASS_PENALTY_RATIO = _pp.get("nev_mass_penalty_ratio", 1.2)

def update_source_settings(new_path, use_camera=False):
    """更新视频源或摄像头配置，并持久化到 config.json"""
    global VIDEO_PATH, USE_CAMERA
    
    # 更新内存
    _cfg["system"]["video_path"] = new_path
    _cfg["system"]["use_camera"] = use_camera
    
    VIDEO_PATH = new_path
    USE_CAMERA = use_camera
    
    # 写入磁盘
    try:
        with open(CONFIG_FILE, "w", encoding='utf-8') as f:
            json.dump(_cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"配置文件写入失败: {e}")
