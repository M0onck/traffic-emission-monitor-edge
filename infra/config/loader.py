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
MASS_FACTOR_EV = _p.get("mass_factor_ev", 1.25)
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

# --- 3. 数据表构建 ---

# 3.1 VSP 系数
VSP_COEFFS = {}
_vsp_raw = _cfg.get("vsp_coefficients", {})
VSP_COEFFS["default"] = _vsp_raw.get("default", {"a_m": 0.156, "b_m": 0.002, "c_m": 0.0005})
for sem_key, coeff_data in _vsp_raw.items():
    if sem_key in VEHICLE_SEMANTIC_MAP:
        VSP_COEFFS[VEHICLE_SEMANTIC_MAP[sem_key]] = coeff_data

# 3.4 车型映射
TYPE_MAP = {}
_type_map_raw = _cfg.get("type_map", {})
for key_str, type_val in _type_map_raw.items():
    if "," in key_str:
        try:
            sem_type, color = key_str.split(",")
            if sem_type.strip() in VEHICLE_SEMANTIC_MAP:
                TYPE_MAP[(VEHICLE_SEMANTIC_MAP[sem_type.strip()], color.strip())] = type_val
        except ValueError: pass
    else:
        TYPE_MAP[key_str] = type_val

# --- 4. 边缘端特有模型路径 ---
_edge = _cfg.get("edge_models", {})
HEF_PATH = _edge.get("hef_path", "resources/yolov8m.hef")
POST_SO_PATH = _edge.get("post_so_path", "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so")
Y5FU_PATH = _edge.get("y5fu_path", "perception/plate_classifier/models/y5fu_320x_sim.onnx")
LITEMODEL_PATH = _edge.get("litemodel_path", "perception/plate_classifier/models/litemodel_cls_96x_r1.onnx")
