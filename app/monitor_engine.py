import cv2
cv2.setNumThreads(1) # 限制OpenCV使用线程数，防止资源争抢
import numpy as np
import supervision as sv
import time
import traceback
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import defaultdict

from ui.renderer import LabelData
from infra.time.ntp_sync import TimeSynchronizer
from domain.physics.spatial_analyzer import SpatialAnalyzer
from domain.physics.kinematics_smoother import KinematicsSmoother
from domain.vehicle.physical_filter import PhysicalVehicleFilter
from domain.physics.vsp_calculator import VSPCalculator
from domain.physics.opmode_mapper import OpModeMapper
from perception.vision_pipeline import VisionPipeline

# 导入感知守护进程
from perception.daemon import perception_worker

logger = logging.getLogger(__name__)

class TrafficMonitorEngine:
    """
    [应用层] 交通监测主引擎 (Traffic Monitor Engine) - 方案 B 多进程版
    """

    def __init__(self, config, components, frame_callback=None):
        self.cfg = config
        self.comps = components
        self.frame_callback = frame_callback
        self._is_running = False
        self.time_sync = TimeSynchronizer()
        
        # ==========================================
        # 1. 进程间通信 (IPC) 核心配置
        # ==========================================
        self.frame_w = self.cfg.FRAME_WIDTH
        self.frame_h = self.cfg.FRAME_HEIGHT
        self.frame_shape = (self.frame_h, self.frame_w, 3)
        self.frame_size = int(np.prod(self.frame_shape))
        
        # 带有时间戳的唯一内存名，防止异常退出导致名称冲突
        self.shm_name = f"nee_cam_shm_{int(time.time())}"
        self.shm = None
        self.shm_array = None
        
        self.p_daemon = None
        self.bbox_queue = mp.Queue(maxsize=3)
        self.stop_event = mp.Event()

        # --- 核心组件引用 ---
        self.registry = components['registry']
        self.db = components['db']
        self.renderer = components['renderer']
        self.plate_worker = components.get('plate_worker', None)

        # --- 物理与业务模型初始化 ---
        self.spatial = SpatialAnalyzer(self.cfg)
        self.kinematics = KinematicsSmoother(self.cfg)
        self.physical_filter = PhysicalVehicleFilter(self.cfg)
        self.vsp_calc = VSPCalculator()
        self.op_mapper = OpModeMapper()
        self.vision_pipeline = VisionPipeline(self.cfg)

        # --- 运行时状态 ---
        self.current_frame_id = 0
        self.active_tracks = {}
        self.current_session_id = None

        # --- 性能分析相关 ---
        self.profile_stats = defaultdict(float)
        self.profile_frames = 0
        self.last_profile_time = time.time()

    def run(self):
        """主引擎运行循环 (消费者模式)"""
        self._is_running = True
        logger.info("[Engine] 正在分配共享内存并启动感知守护进程...")

        try:
            # 1. 开辟操作系统级共享内存
            self.shm = shared_memory.SharedMemory(create=True, size=self.frame_size, name=self.shm_name)
            self.shm_array = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.shm_array[:] = 0 # 初始置黑

            # 2. 启动感知子进程 (隔离 GStreamer 和 NPU)
            self.p_daemon = mp.Process(
                target=perception_worker,
                args=(self.cfg, self.shm_name, self.frame_shape, self.bbox_queue, self.stop_event)
            )
            self.p_daemon.daemon = True
            self.p_daemon.start()

            logger.info("[Engine] 主引擎就绪！系统已进入多进程解耦模式。")

            # 创建新的监测 Session
            session_name = f"AutoSession_{time.strftime('%Y%m%d_%H%M%S')}"
            self.current_session_id = self.db.create_session(
                session_name,
                "Auto started",
                self.cfg.SITE_ELEVATION,
                self.cfg.SITE_GRADIENT
            )
            logger.info(f"[Engine] 创建新监测 Session: {self.current_session_id}")

            last_hailo_data = []

            # 3. 业务主循环
            while self._is_running:
                loop_start = time.perf_counter()

                # 检查子进程是否意外死亡
                if not self.p_daemon.is_alive():
                    logger.error("[Engine] 感知进程意外终止，主引擎停止。")
                    break

                # --- A. 获取隔离数据 ---
                # 瞬间从共享内存拷贝当前帧 (耗时约 1-2 毫秒)
                current_frame = self.shm_array.copy()

                # 非阻塞获取 AI 数据
                if not self.bbox_queue.empty():
                    try:
                        last_hailo_data = self.bbox_queue.get_nowait()
                    except:
                        pass

                # --- B. 业务逻辑处理 ---
                # 防御性检查：确保不是初始化的全黑空白帧
                if current_frame.any():
                    self.current_frame_id += 1
                    self.process_frame(current_frame, last_hailo_data)

                # --- C. 限速保护 ---
                # 限制主进程最高运行在 30fps，防止吃满主核 CPU
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"[Engine] 主循环发生异常: {e}")
            logger.error(traceback.format_exc())
        finally:
            self._cleanup()

    def stop(self):
        """触发停止信号"""
        logger.info("[Engine] 收到停止指令，准备退出...")
        self._is_running = False
        self.stop_event.set()

    def _cleanup(self):
        """资源终极清理：不仅清理业务逻辑，必须清理 IPC 资源"""
        self._is_running = False
        logger.info("[Engine] 开始系统资源与 IPC 内存清理")
        
        # 1. 停止感知进程
        self.stop_event.set()
        if getattr(self, 'p_daemon', None):
            self.p_daemon.join(timeout=4)
            if self.p_daemon.is_alive():
                logger.warning("[Engine] 感知进程未响应，执行强行回收...")
                self.p_daemon.terminate()

        # 2. 回收共享内存 (极其重要，防止内存泄漏)
        if getattr(self, 'shm', None):
            try:
                self.shm.close()
                self.shm.unlink()
                logger.info(f"[Engine] 共享内存 {self.shm_name} 释放完毕。")
            except Exception as e:
                logger.error(f"[Engine] 共享内存释放失败: {e}")

        # 3. 处理历史车辆与轨迹残留
        final_frame_id = getattr(self, 'current_frame_id', 0)
        force_exit_timestamp = time.time() + 1000.0
        self._handle_exits(final_frame_id + 1000, force_exit_timestamp)

        # 4. 回收 OCR 子进程
        if getattr(self, 'plate_worker', None):
            print("[Engine] 正在强制回收 OCR 子进程...")
            self.plate_worker.stop()

        # 5. 宣告采集任务结束
        if getattr(self, 'current_session_id', None):
            self.db.complete_session(self.current_session_id, time.time())
            
        self.db.close()
        logger.info("[Engine] 引擎安全下线。")

    def process_frame(self, frame, hailo_data):
        """处理单帧核心逻辑 (业务逻辑保持原样，不受重构影响)"""
        t0 = time.perf_counter()
        current_time = self.time_sync.get_network_time()
        
        # 1. 目标检测与追踪
        track_t1 = time.perf_counter()
        detections = self.vision_pipeline.process(frame, hailo_data)
        self.profile_stats['01_vision_track'] += (time.perf_counter() - track_t1)

        # 2. ROI 区域过滤
        roi_t1 = time.perf_counter()
        if self.cfg.ENABLE_ROI and self.cfg.ROI_POLYGONS:
            mask = np.zeros(len(detections), dtype=bool)
            for i, bbox in enumerate(detections.xyxy):
                center_point = sv.Position.CENTER.of(bbox)
                for poly_pts in self.cfg.ROI_POLYGONS:
                    if cv2.pointPolygonTest(np.array(poly_pts), (float(center_point[0]), float(center_point[1])), False) >= 0:
                        mask[i] = True
                        break
            detections = detections[mask]
        self.profile_stats['02_roi_filter'] += (time.perf_counter() - roi_t1)
        
        # 3. 车辆运动学与排放计算
        kine_t1 = time.perf_counter()
        self._process_kinematics(detections, current_time)
        self.profile_stats['03_kinematics'] += (time.perf_counter() - kine_t1)

        # 4. 离场处理
        self._handle_exits(self.current_frame_id, current_time)

        # 5. UI 渲染与回调分发
        render_t1 = time.perf_counter()
        if self.frame_callback:
            rendered_frame = self.renderer.render(
                frame.copy(), 
                self.active_tracks,
                self.spatial.camera_calibrator.T if self.spatial.camera_calibrator.is_calibrated else None
            )
            self.frame_callback(rendered_frame)
        self.profile_stats['04_ui_render'] += (time.perf_counter() - render_t1)

        # 总体耗时统计
        self.profile_stats['00_total_process'] += (time.perf_counter() - t0)
        self.profile_frames += 1

        if self.profile_frames >= 30:
            self._print_profile_stats()

    def _process_kinematics(self, detections, current_timestamp):
        """处理当前帧中每个追踪对象的运动学数据"""
        current_track_ids = set()
        
        for tracker_id, bbox, conf, class_id in zip(detections.tracker_id, detections.xyxy, detections.confidence, detections.class_id):
            current_track_ids.add(tracker_id)
            bottom_center = sv.Position.BOTTOM_CENTER.of(bbox)
            
            # --- 构建/更新追踪对象 ---
            if tracker_id not in self.active_tracks:
                vehicle = self.registry.create_vehicle(
                    tracker_id=tracker_id,
                    initial_class_id=class_id,
                    initial_timestamp=current_timestamp
                )
                self.active_tracks[tracker_id] = vehicle
            else:
                vehicle = self.active_tracks[tracker_id]
                vehicle.class_id = class_id # 更新最新类别
                
            vehicle.last_seen = self.current_frame_id
            vehicle.last_timestamp = current_timestamp

            # 送入空间分析器计算物理坐标
            physical_pos = self.spatial.process_point(bottom_center)
            if physical_pos:
                vehicle.add_observation(physical_pos, current_timestamp)

            # 更新运动学参数
            self.kinematics.update_kinematics(vehicle)

            # 物理真实性过滤
            if self.physical_filter.is_valid(vehicle):
                vehicle.is_valid = True
                
                # 计算 VSP 和 OpMode (1Hz 降频计算)
                if len(vehicle.history) >= 2:
                    last_calc_time = getattr(vehicle, 'last_vsp_time', 0)
                    if current_timestamp - last_calc_time >= 1.0: 
                        v_mps = vehicle.current_speed / 3.6
                        a_mps2 = vehicle.current_accel
                        grade = self.cfg.SITE_GRADIENT
                        
                        vsp = self.vsp_calc.calculate_vsp(
                            vehicle_type=self.cfg.CLASS_NAMES.get(vehicle.class_id, "car"),
                            v_mps=v_mps,
                            a_mps2=a_mps2,
                            grade=grade
                        )
                        op_mode = self.op_mapper.get_opmode(
                            vehicle_type=self.cfg.CLASS_NAMES.get(vehicle.class_id, "car"),
                            v_mps=v_mps,
                            a_mps2=a_mps2,
                            vsp=vsp
                        )
                        
                        vehicle.current_vsp = vsp
                        vehicle.current_opmode = op_mode
                        vehicle.last_vsp_time = current_timestamp

            # 送入 OCR 识别车牌 (异步操作，不阻塞主线程)
            if self.plate_worker and vehicle.is_valid and not vehicle.plate_recognized:
                crop = self._crop_vehicle(frame=None, bbox=bbox) # 此处不传入frame，OCR队列会有自身的图像
                if crop is not None:
                    self.plate_worker.enqueue_task(vehicle.id, crop)

    def _crop_vehicle(self, frame, bbox):
        """截取车辆图像供 OCR 使用 (当前仅返回 bbox)"""
        return bbox 
        # TODO: 由于目前OCR逻辑被抽离，此处返回bbox交由外部截取。
        # 在完整架构中，若需要深拷贝图像，请使用 np.ascontiguousarray(frame[y1:y2, x1:x2])

    def _handle_exits(self, current_frame_id, current_timestamp):
        """处理离开画面的车辆"""
        expired_ids = []
        for tid, vehicle in self.active_tracks.items():
            frames_since_last_seen = current_frame_id - vehicle.last_seen
            time_since_last_seen = current_timestamp - vehicle.last_timestamp
            
            if frames_since_last_seen > self.cfg.TRACK_MAX_AGE or time_since_last_seen > 5.0:
                expired_ids.append(tid)
                
        for tid in expired_ids:
            vehicle = self.active_tracks.pop(tid)
            if vehicle.is_valid and self.current_session_id:
                # 车辆离场，保存完整生命周期数据入库
                self._persist_vehicle_data(vehicle)
                self.registry.remove_vehicle(tid)

    def _persist_vehicle_data(self, vehicle):
        """持久化车辆数据"""
        try:
            self.db.save_vehicle_record(
                session_id=self.current_session_id,
                vehicle=vehicle
            )
        except Exception as e:
            logger.error(f"[Engine] 保存车辆 {vehicle.id} 记录失败: {e}")

    def _print_profile_stats(self):
        """打印每 30 帧的性能监控报表"""
        if not getattr(self.cfg, 'ENABLE_PROFILE_LOG', False):
            self.profile_stats.clear()
            self.profile_frames = 0
            return

        log_lines = ["", "="*45, "[性能探针] 过去 30 帧平均耗时 (ms/帧)", "-"*45]
        keys = sorted(self.profile_stats.keys())
        for k in keys:
            avg_ms = (self.profile_stats[k] / self.profile_frames) * 1000
            log_lines.append(f"{k.ljust(20)} : {avg_ms:6.2f} ms")
        
        log_lines.append("-" * 45)
        total_time = time.time() - self.last_profile_time
        fps = self.profile_frames / total_time if total_time > 0 else 0
        log_lines.append(f"-> 实际处理帧率 (FPS): {fps:.2f}")
        log_lines.append("="*45)

        logger.info("\n".join(log_lines))

        self.profile_stats.clear()
        self.profile_frames = 0
        self.last_profile_time = time.time()
