#include <stdint.h>
#include <iostream>
#include "MLX90640_API.h"

// 默认 I2C 地址通常为 0x33
#define MLX_I2C_ADDR 0x33

extern "C" {
    // 静态变量，保证只需初始化一次
    static uint16_t eeMLX90640[832];
    static paramsMLX90640 mlx90640;
    static bool is_initialized = false;

    // 修改1: 将返回值从 void 改为 int，以对接 Python 层的异常捕获机制
    int get_mlx90640_temp(float* temp_array) {
        // 1. 如果是第一次调用，先进行传感器初始化
        if (!is_initialized) {
            // 设置刷新率为 8Hz (0x03)
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0x03);
            MLX90640_SetChessMode(MLX_I2C_ADDR);
            
            // 修改2: 严格校验 EEPROM 读取结果
            int dump_status = MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
            if (dump_status < 0) {
                // 如果读取校准参数失败，直接返回错误码，不要设置 is_initialized
                return dump_status; 
            }
            
            MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
            is_initialized = true;
        }
        
        // 2. 读取一帧原始数据
        uint16_t mlx90640Frame[834];
        int status = MLX90640_GetFrameData(MLX_I2C_ADDR, mlx90640Frame);
        
        // 修改3: 读取失败时，将错误码向上传递给 Python 层，以便触发看门狗重启
        if (status < 0) {
            return status;
        }
        
        // 3. 计算环境参数
        float vdd = MLX90640_GetVdd(mlx90640Frame, &mlx90640);
        float Ta = MLX90640_GetTa(mlx90640Frame, &mlx90640);
        float tr = Ta - 8.0f; // 近似反射温度
        float emissivity = 0.95f; // 地表一般发射率
        
        // 4. 计算绝对温度
        MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, temp_array);

        // 修改4: 成功获取一帧并计算完成后，返回 0
        return 0;
    }
}
