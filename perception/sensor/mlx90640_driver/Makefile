# 编译器设置
CC = g++
# 必须包含 -fPIC 以生成位置无关代码，这是动态库的基础
CXXFLAGS = -O3 -g -fPIC
# SDL2 相关配置仅用于编译可执行程序 Demo
SDL_CFLAGS = `sdl2-config --cflags`
SDL_LFLAGS = `sdl2-config --libs` -lSDL2_ttf

# 路径定义
INCLUDE = -I ./Include -I ./Include/mlx90640-driver
SRC_DIR = ./Source/mlx90640-driver

# 核心驱动源文件 (Linux 平台专用)
DRIVER_SRCS = $(SRC_DIR)/MLX90640_API.cpp $(SRC_DIR)/MLX90640_LINUX_I2C_Driver.cpp
# Python 包装器源文件
WRAPPER_SRC = py_wrapper.cpp

# 目标文件定义
TARGET = main
SO_TARGET = libmlx90640.so

# 告诉 Makefile 在哪里寻找源文件
vpath %.cpp ./Source $(SRC_DIR)

# 默认目标：同时生成 Demo 程序和 Python 动态库
all: $(TARGET) $(SO_TARGET)

# --- 目标 1: 编译 Python 专用的 .so 动态库 ---
# 使用 -shared 参数生成共享库，且不链接 SDL 相关库以保持纯净
$(SO_TARGET): $(DRIVER_SRCS) $(WRAPPER_SRC)
	$(CC) -shared $(CXXFLAGS) $(INCLUDE) $^ -o $@

# --- 目标 2: 编译原有的 C++ 可执行程序 Demo ---
$(TARGET): main.o MLX90640_API.o MLX90640_LINUX_I2C_Driver.o
	$(CC) $(CXXFLAGS) $(INCLUDE) $^ -o $@ $(SDL_LFLAGS)

# --- 中间编译规则 ---
main.o: main.cpp
	$(CC) $(CXXFLAGS) $(SDL_CFLAGS) $(INCLUDE) -c $< -o $@

MLX90640_API.o: $(SRC_DIR)/MLX90640_API.cpp
	$(CC) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

MLX90640_LINUX_I2C_Driver.o: $(SRC_DIR)/MLX90640_LINUX_I2C_Driver.cpp
	$(CC) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# 清理规则
clean:
	rm -f *.o *.a $(TARGET) $(SO_TARGET)
