import numpy as np
import time
import logging
import threading
from multiprocessing import shared_memory
# 【导入原生 HailoRT】
from hailo_platform import (VDevice, HEF, ConfigureParams, FormatType, HailoStreamInterface, InputVStreamParams, OutputVStreamParams)

logger = logging.getLogger(__name__)

def perception_worker(shm_name, shape, bbox_queue, stop_event, config_dict):
    # 1. 启动剥离了 AI 插件的纯 GStreamer 流水线
    frame_ready_event = threading.Event()
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shm_array = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)
    
    pipeline = GstPipelineManager(config_dict, shm_array=shm_array, frame_ready_event=frame_ready_event)
    pipeline.start()

    # 2. 【初始化原生 Hailo 硬件】
    hef_path = config_dict.get("hef_path", "resources/yolov8m.hef")
    
    try:
        # 创建虚拟设备并加载模型
        target = VDevice()
        hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        # 配置输入输出流的格式（输入 UINT8 RGB，输出 FLOAT32）
        input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        logger.info("-> [感知进程] PyHailoRT 原生推理引擎初始化成功！")

        # 3. 激活网络并开启数据流
        with network_group.activate(network_group_params):
            with target.get_input_vstreams(input_vstream_params) as input_vstreams, \
                 target.get_output_vstreams(output_vstream_params) as output_vstreams:
                 
                in_vstream = input_vstreams[0]
                 
                while not stop_event.is_set():
                    # 死等摄像头出图（硬件起搏器）
                    if frame_ready_event.wait(timeout=0.2):
                        frame_ready_event.clear()
                        
                        ai_frame = pipeline.ai_frame # 获取 640x640 画面
                        if ai_frame is None:
                            continue

                        # ====================================================
                        # 【绝对安全的同步推理】: 送入 NPU，死等结果，彻底隔绝竞态！
                        # ====================================================
                        # 注意：Hailo 要求输入 shape 为 (1, 640, 640, 3) 且连续
                        input_data = np.expand_dims(ai_frame, axis=0).astype(np.uint8)
                        in_vstream.send(input_data)
                        
                        raw_outputs = []
                        for out_stream in output_vstreams:
                            raw_outputs.append(out_stream.recv())
                        # ====================================================

                        # TODO: 此时 raw_outputs 是 NPU 输出的张量矩阵 (Numpy array)
                        # 你需要在这里调用 Python 版的 YOLO 后处理 (NMS 非极大值抑制)
                        hailo_data = custom_yolov8_postprocess(raw_outputs)
                        
                        # 推送给主进程
                        while not bbox_queue.empty():
                            try: bbox_queue.get_nowait()
                            except: pass
                        try: bbox_queue.put_nowait(hailo_data)
                        except: pass

    except Exception as e:
        logger.error(f"[PyHailoRT] 引擎发生致命错误: {e}", exc_info=True)
    finally:
        pipeline.stop()
        # VDevice 等底层 C++ 资源会随 Context Manager 自动销毁
