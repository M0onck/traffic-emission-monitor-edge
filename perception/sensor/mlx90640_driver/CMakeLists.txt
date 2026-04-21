# 明确指定只编译 Linux 平台需要的核心 API 和 LINUX I2C 驱动
set(MLX_SOURCES 
    Source/mlx90640-driver/MLX90640_API.cpp
    Source/mlx90640-driver/MLX90640_LINUX_I2C_Driver.cpp
)

# 生成动态链接库
add_library(mlx90640_driver SHARED ${MLX_SOURCES} py_wrapper.cpp)

# 指定头文件包含路径（补全原 Makefile 中的两个 Include 路径）
target_include_directories(mlx90640_driver PRIVATE 
    Include 
    Include/mlx90640-driver
)
