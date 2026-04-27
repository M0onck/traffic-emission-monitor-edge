import os
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import signal
import logging
import threading
from perception.gst_pipeline import GstPipelineManager

# 导入原生 PyHailoRT
from hailo_platform import (
    VDevice, HEF, ConfigureParams, FormatType, 
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
    InferVStreams
)

logger = logging.getLogger(__name__)

def parse_hailo_ragged_list(raw_list, conf_threshold=0.45):
    """提取 Hailo 硬件 NMS 输出，保持归一化坐标供 VisionPipeline 消费"""
    results = []
    batch_0_data = raw_list[0] 
    
    # YOLOv8 官方 COCO 数据集类别映射
    # 2: car (小汽车), 5: bus (公交车), 7: truck (卡车)
    coco_map = {2: "car", 5: "bus", 7: "truck"}
    
    for class_id, boxes in enumerate(batch_0_data):
        # 仅放行字典中我们关心的车辆类别
        if class_id not in coco_map:
            continue
            
        label_str = coco_map[class_id]
        
        for box in boxes:
            ymin, xmin, ymax, xmax, score = box
            if score < conf_threshold: 
                continue
            
            # 原封不动地返回 0.0~1.0 的归一化坐标
            results.append({
                "xmin": float(xmin), 
                "ymin": float(ymin), 
                "xmax": float(xmax), 
                "ymax": float(ymax),
                "conf": float(score), 
                "label": label_str  # 正确传递分类字符串
            })
            
    return results

def perception_worker(shm_name, shape, bbox_queue, stop_event, config_dict, ready_event):
    logger.info("-> [感知进程] 启动，准备挂载 GStreamer 与 PyHailoRT...")

    # 捕获看门狗发出的 SIGTERM 信号，触发优雅退出
    def sigterm_handler(signum, frame):
        logger.warning(f"[感知进程] 收到终止信号 ({signum})，正在触发安全停止，保存视频封口...")
        stop_event.set()

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler) # 兼容终端 Ctrl+C 强制结束

    class ConfigWrapper:
        def __init__(self, d):
            for k, v in d.items(): setattr(self, k, v)

    runtime_config = ConfigWrapper(config_dict)
    
    pipeline = None
    target = None
    try:
        # 1. 初始化 IPC 机制与 GStreamer (提供画面)
        frame_ready_event = threading.Event()
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shm_array = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)
        
        pipeline = GstPipelineManager(runtime_config, shm_array=shm_array, frame_ready_event=frame_ready_event)
        pipeline.start()

        # 2. 初始化 PyHailoRT (提供算力)
        hef_path = config_dict.get("hef_path", "resources/yolov8m.hef")
        target = VDevice()
        hef = HEF(hef_path)
        
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name

        logger.info("-> [感知进程] 底层硬件环境部署完毕，开始进入主循环...")

        # 3. 激活硬件推理管线并进入主循环
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):

                # 通知主进程看门狗，子线程已初始化完毕
                if ready_event:
                    ready_event.set()
                
                while not stop_event.is_set():
                    # 等待 GStreamer 的新画面
                    if frame_ready_event.wait(timeout=0.2):
                        frame_ready_event.clear()
                        
                        ai_frame = pipeline.ai_frame
                        if ai_frame is None:
                            continue

                        # A. 组织输入数据 (扩展 batch 维)
                        input_data = np.expand_dims(ai_frame, axis=0)
                        
                        # B. 安全的同步推理
                        infer_results = infer_pipeline.infer({input_name: input_data})
                        
                        # C. 提取与压铸
                        raw_output = infer_results[output_name]
                        
                        # D. 极速后处理
                        hailo_data = parse_hailo_ragged_list(raw_output, conf_threshold=0.45)
                        
                        # E. IPC 推送给主引擎
                        while not bbox_queue.empty():
                            try: bbox_queue.get_nowait()
                            except: pass
                        try: bbox_queue.put_nowait(hailo_data)
                        except: pass

    except KeyboardInterrupt:
        logger.info("[感知进程] 收到中断信号。")
    except Exception as e:
        logger.error(f"[感知进程] 发生异常: {e}", exc_info=True)
    finally:
        logger.info("[感知进程] 正在清理资源...")
        if pipeline:
            pipeline.stop()
        if target:
            target.release() # 确保 VDevice 彻底释放硬件锁
        try:
            existing_shm.close()
        except: pass

        logger.info("[感知进程] 资源清理完毕，进程安全退出。")
