# perception/sensor/hailo_bridge/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(HailoBridge VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==========================================
# 目标输出路径：直接扔进工程根目录的 bin 文件夹
# ==========================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../bin)

# ==========================================
# 1. 链接 GStreamer 基础库
# ==========================================
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

add_library(hailo_bridge SHARED libhailo_bridge.cpp)
target_link_libraries(hailo_bridge PRIVATE PkgConfig::GST)

# ==========================================
# 2. 精确链接 Hailo 树莓派专属架构 (基于 dpkg 扫描结果)
# ==========================================
message(STATUS "---------------------------------------------------")
message(STATUS "[Hailo Bridge] 正在挂载树莓派专属 TAPPAS 架构...")

# 引入头文件目录
target_include_directories(hailo_bridge PRIVATE /usr/include/hailo/tappas)

# 直接锁定目标动态库：gsthailometa
find_library(HAILO_META_LIB 
    NAMES gsthailometa 
    PATHS /usr/lib/aarch64-linux-gnu 
    REQUIRED
)

if(HAILO_META_LIB)
    message(STATUS "[Hailo Bridge] 成功锁定核心元数据库: ${HAILO_META_LIB}")
    target_link_libraries(hailo_bridge PRIVATE ${HAILO_META_LIB})
endif()

message(STATUS "---------------------------------------------------")
