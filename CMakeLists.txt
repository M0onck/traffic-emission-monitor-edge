cmake_minimum_required(VERSION 3.15)
project(TrafficEmissionMonitor LANGUAGES CXX C)

# 全局 C++ 标准与编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 强制生成位置无关代码 (-fPIC)，用于构建 .so

# 统一输出目录：所有编译出的 .so 文件将集中存放在项目根目录的 build/lib 下
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 查找 GStreamer 相关依赖
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-gl-1.0 gstreamer-base-1.0)

# 挂载子模块
add_subdirectory(perception/sensor/mlx90640_driver)
add_subdirectory(perception/sensor/weather_driver)
add_subdirectory(plugins/dewarp)
