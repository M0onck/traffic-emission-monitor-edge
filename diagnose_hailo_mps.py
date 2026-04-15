import numpy as np
import time

def run_test(mode_name, use_mps, force_copy):
    print(f"\n{'='*50}")
    print(f"🧪 测试模式: {mode_name}")
    print(f"   - 启用 MPS 共享: {use_mps}")
    print(f"   - 强制深拷贝内存: {force_copy}")
    print(f"{'='*50}")
    
    try:
        from hailo_platform import VDevice, HEF, InferVStreams, InputVStreamParams, OutputVStreamParams, HailoSchedulingAlgorithm
    except ImportError:
        print("❌ 无法导入 hailo_platform")
        return

    params = VDevice.create_params()
    if use_mps:
        params.group_id = "SHARED"
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.multi_process_service = True

    try:
        with VDevice(params) as vdevice:
            hef = HEF("perception/plate_classifier/models/y5fu.hef")
            input_info = hef.get_input_vstream_infos()[0]
            in_name = input_info.name
            
            network_group = vdevice.configure(hef)[0]
            in_params = InputVStreamParams.make(network_group)
            out_params = OutputVStreamParams.make(network_group)
            
            with InferVStreams(network_group, in_params, out_params) as infer_pipeline:
                # 1. 直接构造标准的 4D 数组
                tensor_4d = np.full((1, 320, 320, 3), 255, dtype=np.uint8)
                
                # 2. 核心变量：是否强制深拷贝（解决底层指针丢失问题）
                if force_copy:
                    tensor_4d = np.array(tensor_4d, copy=True)
                else:
                    tensor_4d = np.ascontiguousarray(tensor_4d)
                    
                print(f"📦 内存状态: OWNDATA={tensor_4d.flags['OWNDATA']}, C_CONTIGUOUS={tensor_4d.flags['C_CONTIGUOUS']}")
                
                # 3. 灌入数据
                result = infer_pipeline.infer({in_name: tensor_4d})
                print("✅✅✅ 推理成功！底层 C++ 完美接收到了数据！")
                
    except Exception as e:
        err_msg = str(e).split('See hailort.log')[0].strip()
        print(f"❌ 崩溃报错: {err_msg}")

if __name__ == "__main__":
    print("🚨 警告：请确保你已经使用 Ctrl+C 彻底关闭了包含 GStreamer 的主程序！")
    time.sleep(2)
    
    # 测试 1: MPS模式 + 强制深拷贝 
    # (验证是否是 Python 内存视图导致 got 0)
    run_test("测试 1 (MPS + 内存深拷贝)", use_mps=True, force_copy=True)
    
    # 测试 2: 独占模式 + 强制深拷贝 
    # (验证是否是 MPS 守护进程本身的同步 Bug)
    run_test("测试 2 (独占模式 + 内存深拷贝)", use_mps=False, force_copy=True)
