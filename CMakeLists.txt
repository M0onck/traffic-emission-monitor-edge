# traffic-emission-monitor-edge/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(TrafficEmissionMonitor LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 强制生成位置无关代码 (-fPIC)

# 重定向所有的输出产物
# 动态库 (.so) 输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# 静态库 (.a) 输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# 可执行文件 (.out) 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 为了兼容不同的生成器（如 Ninja/Make 等），将所有配置都指向同一个 lib 目录
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

# 查找 GStreamer 相关依赖
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-gl-1.0 gstreamer-base-1.0)

# 挂载子模块（这里必须在上述 set 变量之后执行）
add_subdirectory(perception/sensor/mlx90640_driver)
add_subdirectory(perception/sensor/weather_driver)
add_subdirectory(plugins/dewarp)
