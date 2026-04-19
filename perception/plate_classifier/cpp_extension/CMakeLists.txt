cmake_minimum_required(VERSION 3.10)
project(hailo_ocr_cpp)

set(CMAKE_CXX_STANDARD 17)

# 寻找依赖包
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(hailort REQUIRED) # 第一步先注释掉，跑通基础流后再加

# 创建 Python 模块
pybind11_add_module(hailo_ocr_cpp 
    src/bindings.cpp 
    src/plate_recognizer.cpp
)

target_include_directories(hailo_ocr_cpp PRIVATE include)
target_link_libraries(hailo_ocr_cpp PRIVATE ${OpenCV_LIBS})

target_link_libraries(hailo_ocr_cpp PRIVATE ${OpenCV_LIBS} hailort)
