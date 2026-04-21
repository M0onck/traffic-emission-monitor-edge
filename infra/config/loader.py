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
RUN_MODE = _sys.get("run_mode", "inference").lower() # 'inference' 或 'collection'
VIDEO_PATH = _sys["video_path"]
LOCAL_VIDEO_PATH = _sys.get("local_video_path", VIDEO_PATH)
DB_PATH = _sys["db_path"]
FPS = max(1.0, float(_sys.get("fps", 30.0))) # [防御性编程] 防止ZeroDivisionError
DEBUG_MODE = _sys["debug_mode"]
USE_CAMERA = _sys.get("use_camera", False)

_lib = _cfg["lib_paths"]
WS_PATH = _lib.get("libweather_path", "build/lib/libweather_driver.so")
TC_PATH = _lib.get("libthermal_path", "build/lib/libmlx90640_driver.so")

_rec = _cfg.get("record_options", {})
ENABLE_RECORD = _rec.get("enable_record", False)
RECORD_SEGMENT_MIN = _rec.get("segment_length_min", 10)
RECORD_SAVE_PATH = _rec.get("save_path", "/mnt/nvmessd/recorded_videos") # 默认优先用 SSD

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
EXIT_TIMEOUT_SEC = _k.get("exit_timeout_sec", 1.0)
KINEMATICS_POLY_ORDER = _k.get("poly_order", 3)

_o = _cfg["ocr_params"]
OCR_RETRY_COOLDOWN = _o["retry_cooldown"]
OCR_CONF_THRESHOLD = _o["confidence_threshold"]

_p = _cfg.get("physics_params", {})
ROAD_GRADE_PERCENT = _p.get("road_grade_percent", 0.0)

# 质量控制参数
_qc = _cfg.get("quality_control", {})
MIN_VALID_POINTS = _qc.get("min_valid_trajectory_points", 15)
MIN_SURVIVAL_SEC = _qc.get("min_survival_sec", 1.0)
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

# [防御性编程] 为所有时序窗口强制施加物理/性能安全底线
# 1. 强制延迟 > 30秒: 确保绝大多数车辆有足够的时间穿过监控画面并完成入库
ALIGNMENT_DELAY_SEC = max(30.0, float(_tw.get("alignment_delay_sec", 60.0)))

# 2. 强制积分窗 > 60秒: 积分窗口如果太短，VSP 能量累积会失去统计学意义
INTEGRATION_WINDOW_SEC = max(60.0, float(_tw.get("integration_window_sec", 300.0)))

# 3. 强制基线窗 > 5分钟: 保证至少有 5 分钟的跨度去寻找干净的 PMC 极小值
BASELINE_WINDOW_MINUTE = max(5.0, float(_tw.get("baseline_window_minute", 10.0)))

# 4. 强制对齐步长 > 10秒: 限制 SQLite 的密集 I/O 频率，防止树莓派 CPU 顶不住
DB_ALIGN_INTERVAL_SEC = max(10.0, float(_tw.get("db_align_interval_sec", 60.0)))

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

    # 只有在使用本地源时，才更新并保存 local_video_path
    if not use_camera:
        _cfg["system"]["local_video_path"] = new_path
        LOCAL_VIDEO_PATH = new_path
    
    # 写入磁盘
    try:
        with open(CONFIG_FILE, "w", encoding='utf-8') as f:
            json.dump(_cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"配置文件写入失败: {e}")

def update_run_mode(new_mode):
    """更新系统运行模式 (inference/collection) 并持久化"""
    global RUN_MODE
    new_mode = new_mode.lower()
    if new_mode in ['inference', 'collection']:
        _cfg["system"]["run_mode"] = new_mode
        RUN_MODE = new_mode
        try:
            with open(CONFIG_FILE, "w", encoding='utf-8') as f:
                json.dump(_cfg, f, indent=2, ensure_ascii=False)
            print(f"[Config] 运行模式已切换为: {new_mode}")
        except Exception as e:
            print(f"配置文件写入失败: {e}")

def update_record_settings(enable: bool, segment_min: int, path: str):
    """更新视频录制选项并持久化到 config.json"""
    global ENABLE_RECORD, RECORD_SEGMENT_MIN, RECORD_SAVE_PATH
    
    if "record_options" not in _cfg:
        _cfg["record_options"] = {}
        
    _cfg["record_options"]["enable_record"] = enable
    _cfg["record_options"]["segment_length_min"] = segment_min
    _cfg["record_options"]["save_path"] = path

    ENABLE_RECORD = enable
    RECORD_SEGMENT_MIN = segment_min
    RECORD_SAVE_PATH = path

    try:
        with open(CONFIG_FILE, "w", encoding='utf-8') as f:
            json.dump(_cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"配置文件写入失败: {e}")
