# app/bootstrap.py
import os
import sys
import subprocess
import logging
import numpy as np
import multiprocessing as mp
from infra.store.sqlite_manager import DatabaseManager
from infra.concurrency.async_recognizer import AsyncPlateRecognizer
from domain.vehicle.repository import VehicleRegistry
from domain.vehicle.classifier import VehicleClassifier
from infra.store.storage_manager import StorageManager
from perception.sensor.thermal_camera import ThermalCamera
from perception.sensor.weather_station import WeatherGateway
from ui.renderer import Visualizer
from app.alignment_daemon import AlignmentDaemon

logger = logging.getLogger(__name__)

def sync_native_extensions():
    """
    统一管理并按需编译项目中所有的 C++ 扩展 (传感器驱动、GStreamer 插件)
    """
    # 动态获取项目根目录 (根据 app/bootstrap.py 向上推两级)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, "build")
    lib_dir = os.path.join(build_dir, "bin")
    
    logger.info("[Bootstrap] 正在检查并同步 C++ 扩展模块...")
    
    try:
        os.makedirs(build_dir, exist_ok=True)
        
        # 1. 运行 CMake 配置 (CMake 会自动处理缓存和依赖关系)
        subprocess.run(
            ["cmake", ".."], 
            cwd=build_dir, 
            check=True, 
            capture_output=True
        )
        
        # 2. 运行 Make 并发编译 (树莓派 5 推荐使用 -j4)
        subprocess.run(
            ["make", "-j4"], 
            cwd=build_dir, 
            check=True,
            capture_output=False # 如果不想看满屏编译日志可屏蔽，想看可设为 False
        )
        
        # 3. 注入 GStreamer 环境变量，使其能够发现刚刚编译的 dewarp 插件
        current_gst_path = os.environ.get("GST_PLUGIN_PATH", "")
        os.environ["GST_PLUGIN_PATH"] = f"{lib_dir}:{current_gst_path}"
        
        logger.info(f"[Bootstrap] C++ 扩展同步就绪。共享库目录: {lib_dir}")
        return lib_dir
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[Bootstrap] C++ 模块编译失败！返回码: {e.returncode}")
        if e.stderr:
            logger.error(f"编译错误详情: {e.stderr.decode('utf-8')}")
        sys.exit(1)

class AppBootstrap:
    """
    [应用引导程序] 负责系统组件的实例化、依赖注入与多进程 IPC 通道建立。
    """
    
    @staticmethod
    def _run_alignment_worker(sync_queue, stop_event):
        # 在新进程内部导入配置文件
        # 这样每次子进程苏醒时，都会自动获得属于自己的有效 cfg 对象
        import infra.config.loader as cfg 
        from app.alignment_daemon import AlignmentDaemon

        # 将刚刚导入的 cfg 传给守护进程
        daemon = AlignmentDaemon(cfg, sync_queue, stop_event) 
        daemon.run()

    @staticmethod
    def setup_components(config):
        print(">>> [Bootstrap] 正在组装系统组件与 IPC 通道...")
        
        # --- 核心新增：全局通信管道与控制信号 ---
        ctx = mp.get_context('spawn')
        sync_queue = ctx.Queue(maxsize=10) # 限制容量防内存溢出
        stop_event = ctx.Event()

        # 1. 基础设施层
        StorageManager.ensure_structure()
        db = DatabaseManager(db_path=config.DB_PATH, fps=config.FPS)
        alignment_delay = getattr(config, 'ALIGNMENT_DELAY_SEC', 60.0)
        # 强制落盘时间必须小于延迟窗口，留出安全期
        # 保证对齐引擎回头查数据时，车辆数据已经在数据库里了
        force_delay = max(3.0, alignment_delay - 3.0)

        # 2. 气象站实例
        try:
            weather_gw = WeatherGateway(getattr(config, 'WS_PATH', 'build/lib/libmlx90640_driver.so'))
            weather_gw.start()
            print(">>> [Bootstrap] 气象站驱动已拉起。")
        except Exception as e:
            print(f">>> [Bootstrap] 气象驱动加载失败: {e}")
            weather_gw = None

        # 3. 领域层
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

        # 4. 异步处理与传感器
        plate_worker = AsyncPlateRecognizer() if getattr(config, 'ENABLE_OCR', False) else None
        thermal_cam = ThermalCamera(getattr(config, 'TC_PATH', 'build/lib/libmlx90640_driver.so'))

        # 5. 渲染层
        target_pts_raw = getattr(config, 'TARGET_POINTS', [[0,0], [1,0], [1,1], [0,1]])
        target_points = np.array(target_pts_raw, dtype=np.float32)
        norm_source_points = np.array(config.SOURCE_POINTS, dtype=np.float32) if getattr(config, 'SOURCE_POINTS', None) else None
        visualizer = Visualizer(calibration_points=target_points, target_fps=config.FPS)

        # --- 启动后台对齐进程 ---
        # 无论当前是采集模式还是推理模式，只要系统拉起，就无条件启动 L2 级快照生产线
        alignment_proc = ctx.Process(
            target=AppBootstrap._run_alignment_worker,
            args=(sync_queue, stop_event),
            daemon=True
        )
        alignment_proc.start() # 在此处启动，由于是阻塞队列，它会安静地等待主引擎的 tick
        print(">>> [Bootstrap] 延迟对齐后台进程已挂载。")

        # 6. 封装最终字典
        components = {
            'config': config,
            'weather_station': weather_gw,
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
