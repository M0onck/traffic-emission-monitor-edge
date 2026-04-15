import multiprocessing as mp
import time
import sys

def hailo_worker(worker_id):
    """子进程任务：尝试以共享模式挂载 NPU 并保持一段时间"""
    try:
        from hailo_platform import VDevice
    except ImportError:
        print(f"[Worker {worker_id}] ❌ 无法导入 hailo_platform，请检查虚拟环境。")
        sys.exit(1)

    # 1. 创建 VDevice 参数并指定共享组
    params = VDevice.create_params()
    params.group_id = "SHARED"

    from hailo_platform import HailoSchedulingAlgorithm
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.multi_process_service = True  # 显式告诉底层调用 hailort.service，而不是直连物理硬件
    
    print(f"⏳ [Worker {worker_id}] 正在尝试连接 NPU (加入 SHARED 组)...")
    
    try:
        # 2. 尝试挂载 NPU
        with VDevice(params) as vdevice:
            print(f"✅ [Worker {worker_id}] 成功挂载 NPU！已进入共享池。")
            
            # 3. 故意休眠几秒钟，保持硬件锁定状态
            # 这能确保后面的进程启动时，当前进程仍然占有着 NPU
            for i in range(3):
                print(f"   [Worker {worker_id}] 正在使用 NPU 模拟推理任务 {i+1}/3...")
                time.sleep(1)
                
            print(f"🏁 [Worker {worker_id}] 任务完成，正常断开连接。")
            
    except Exception as e:
        print(f"❌ [Worker {worker_id}] 连接失败，抛出异常:\n{e}")

if __name__ == '__main__':
    print("========== Hailo 多进程共享 (MPS) 测试开始 ==========")
    
    # 我们同时孵化 2 个子进程，模拟主视觉管道和异步 OCR 管道争抢资源的情况
    num_workers = 2
    processes = []
    
    for i in range(num_workers):
        p = mp.Process(target=hailo_worker, args=(i+1,))
        processes.append(p)
        p.start()
        # 故意错开 0.5 秒启动，模拟真实场景中进程初始化的先后顺序
        time.sleep(0.5) 
        
    for p in processes:
        p.join()
        
    print("========== 测试结束 ==========")
