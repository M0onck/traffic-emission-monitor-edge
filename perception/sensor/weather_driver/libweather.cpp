/**
 * @file weather_monitor.cpp
 * @brief ESP32 气象站远程监控终端 (PC 上位机)
 * @version 3.2.0 (Linux Blocking-IO Fix)
 * @date 2026-02-07
 * @author User & AI Assistant
 * @details
 * 修复日志 (Linux 平台):
 * 1. 修正 openPort 中错误清除 O_NDELAY 标志导致串口进入阻塞模式的问题。
 * (解决发送端断电后，上位机UI卡死不更新状态的 Bug)
 * 2. 增强 readData 的错误处理，正确区分 "EAGAIN(空闲)" 与 "EIO/EOF(拔出)"。
 * (解决拔出设备后无法触发自动重连的 Bug)
 */

#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <sstream>
#include <ctime>

// --- OS-Specific Headers ---
#ifdef _WIN32
    #include <windows.h>
    #include <tchar.h>
#else
    #include <fcntl.h>
    #include <termios.h>
    #include <unistd.h>
    #include <dirent.h>
    #include <cerrno> // 关键：用于访问 errno
#endif

// ==========================================
//               配置常量定义
// ==========================================

/// @brief 串口波特率
const int  BAUD_RATE        = 115200;

/// @brief 握手阶段超时 (毫秒)
const int  SCAN_TIMEOUT_MS  = 300;

/// @brief 数据看门狗超时 (毫秒)
const int  DATA_TIMEOUT_MS  = 3000;

const std::string CMD_HANDSHAKE = "CMD:WHO_ARE_YOU\n"; 
const std::string ACK_HANDSHAKE = "ACK:WEATHER_RX_V1"; 
const std::string DATA_PREFIX   = "DATA:";             

// ==========================================
//               UI 样式定义 (ANSI)
// ==========================================
const char* TERM_CLEAR_HOME = "\033[2J\033[H"; 
const char* TERM_GREEN      = "\033[32m";      
const char* TERM_RED        = "\033[31m";      
const char* TERM_YELLOW     = "\033[33m";      
const char* TERM_RESET      = "\033[0m";       

/**
 * @class SerialPort
 * @brief 跨平台串口操作封装类
 */
class SerialPort {
private:
#ifdef _WIN32
    HANDLE hSerial;
#else
    int fd;
#endif
    bool connected;
    std::string portName;

public:
    SerialPort() : connected(false) {
#ifdef _WIN32
        hSerial = INVALID_HANDLE_VALUE;
#else
        fd = -1;
#endif
    }
    
    ~SerialPort() { closePort(); }

    /**
     * @brief 打开串口并配置参数
     */
    bool openPort(std::string port, int baudRate) {
        portName = port;

#ifdef _WIN32
        std::string fullPortName = (port.length() > 4) ? "\\\\.\\" + port : port;
        hSerial = CreateFileA(fullPortName.c_str(), GENERIC_READ | GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        if (hSerial == INVALID_HANDLE_VALUE) return false;

        COMMTIMEOUTS timeouts = {0};
        timeouts.ReadIntervalTimeout = 50;
        timeouts.ReadTotalTimeoutConstant = 50;
        timeouts.ReadTotalTimeoutMultiplier = 10;
        if (!SetCommTimeouts(hSerial, &timeouts)) { CloseHandle(hSerial); return false; }

        DCB dcb = {0};
        dcb.DCBlength = sizeof(dcb);
        if (!GetCommState(hSerial, &dcb)) { CloseHandle(hSerial); return false; }
        
        dcb.BaudRate = baudRate;
        dcb.ByteSize = 8;
        dcb.StopBits = ONESTOPBIT;
        dcb.Parity = NOPARITY;
        dcb.fDtrControl = DTR_CONTROL_DISABLE; 
        dcb.fRtsControl = RTS_CONTROL_DISABLE; 
        
        if (!SetCommState(hSerial, &dcb)) { CloseHandle(hSerial); return false; }

        EscapeCommFunction(hSerial, SETDTR); 
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        EscapeCommFunction(hSerial, CLRDTR); 
        PurgeComm(hSerial, PURGE_RXCLEAR | PURGE_TXCLEAR);

#else
        // Linux: 核心修复部分
        
        // 1. O_RDWR: 读写模式
        // 2. O_NOCTTY: 不将此端口作为控制终端 (防止 Ctrl+C 杀掉程序)
        // 3. O_NONBLOCK: 关键！非阻塞模式，确保 read() 不会卡死
        fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd == -1) return false;
        
        struct termios options;
        if (tcgetattr(fd, &options) != 0) { close(fd); return false; }

        cfsetispeed(&options, B115200);
        cfsetospeed(&options, B115200);
        
        // 配置 Raw Mode (原始模式)
        options.c_cflag |= (CLOCAL | CREAD); // 忽略 Modem 控制线，启用接收
        options.c_cflag &= ~PARENB;          // 无校验
        options.c_cflag &= ~CSTOPB;          // 1 停止位
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;              // 8 数据位
        
        // 禁用硬件流控 (RTS/CTS) - 对于 ESP32 很重要
        options.c_cflag &= ~CRTSCTS;

        // 禁用软件流控
        options.c_iflag &= ~(IXON | IXOFF | IXANY);
        
        // 设置为 Raw 输入 (禁用回显、信号处理等)
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); 
        options.c_oflag &= ~OPOST; 

        // 关键修复：不要使用 fcntl(fd, F_SETFL, 0) !
        // 之前的代码在这里清除了 O_NONBLOCK，导致变成了阻塞模式。
        // 我们保持 open 时设置的标志不变。

        tcsetattr(fd, TCSANOW, &options);
        
        // 清空缓冲区
        tcflush(fd, TCIOFLUSH);
#endif

        connected = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); 
        return true;
    }

    /**
     * @brief 关闭串口
     */
    void closePort() {
        if (!connected) return;
#ifdef _WIN32
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
#else
        close(fd);
        fd = -1;
#endif
        connected = false;
    }

    /**
     * @brief 写入数据
     */
    bool writeData(const std::string& data) {
        if (!connected) return false;
#ifdef _WIN32
        DWORD bytesWritten;
        return WriteFile(hSerial, data.c_str(), data.length(), &bytesWritten, NULL);
#else
        return write(fd, data.c_str(), data.length()) > 0;
#endif
    }

    /**
     * @brief 读取数据 (Linux 修复版)
     * @return int 
     * > 0: 读取字节数
     * 0:  空闲 (无数据)
     * -1: 硬件错误 (断开)
     */
    int readData(char* buffer, int bufSize) {
        if (!connected) return -1;
#ifdef _WIN32
        DWORD bytesRead = 0;
        if (ReadFile(hSerial, buffer, bufSize, &bytesRead, NULL)) {
            return (int)bytesRead;
        }
        return -1; 
#else
        int bytesRead = read(fd, buffer, bufSize);
        
        // 情况 1: 读到数据
        if (bytesRead > 0) return bytesRead;
        
        // 情况 2: read 返回 0
        // 在串口编程中，read 返回 0 通常意味着 EOF (挂起)，即设备物理断开
        if (bytesRead == 0) return -1; 
        
        // 情况 3: read 返回 -1 (错误)
        if (bytesRead < 0) {
            // 如果是 EAGAIN (Try again) 或 EWOULDBLOCK
            // 说明是非阻塞模式下当前没有数据，这是正常状态
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return 0; // 返回 0 表示空闲，交给上层去判断超时
            }
            // 其他错误 (如 EIO, ENODEV, EBADF) 均视为硬件故障
            return -1; 
        }
        return -1;
#endif
    }

    bool isConnected() { return connected; }
    std::string getName() { return portName; }
};

/**
 * @brief 获取可用串口列表
 */
std::vector<std::string> getAvailablePorts() {
    std::vector<std::string> ports;
#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DEVICEMAP\\SERIALCOMM", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char valueName[256];
        char data[256];
        DWORD valueNameSize, dataSize, type, index = 0;
        while (true) {
            valueNameSize = sizeof(valueName);
            dataSize = sizeof(data);
            if (RegEnumValueA(hKey, index++, valueName, &valueNameSize, NULL, &type, (LPBYTE)data, &dataSize) != ERROR_SUCCESS) break;
            std::string k = valueName;
            if (k.find("Bth") == std::string::npos && k.find("Bluetooth") == std::string::npos) 
                ports.push_back(data);
        }
        RegCloseKey(hKey);
    } 
#else
    DIR *dir; struct dirent *ent;
    if ((dir = opendir("/dev/")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string name = ent->d_name;
            // 扫描 ttyUSB (USB转串口) 和 ttyACM (CDC类串口，常见于ESP32-S3/C3)
            if (name.find("ttyUSB") != std::string::npos || name.find("ttyACM") != std::string::npos) 
                ports.push_back("/dev/" + name);
        }
        closedir(dir);
    }
#endif
    return ports;
}

// ==========================================
//    供 Python 调用的 C 结构体 (需严格内存对齐)
// ==========================================
struct WeatherDataC {
    float temp;
    float humidity;
    float windSpeed;
    int windDir;
    int pm25;
    int pm10;
    uint32_t timestamp;
    bool isOnline;
};

// 全局状态与线程锁
std::mutex data_mtx;
WeatherDataC shared_data = {0, 0, 0, 0, 0, 0, 0, false};
std::atomic<bool> is_running(false);
std::thread* bg_thread = nullptr;
SerialPort* global_serial = nullptr;

// 解析逗号分隔的数据串并存入结构体
void parseLineToStruct(const std::string& line) {
    if (line.find(DATA_PREFIX) == 0) {
        std::string content = line.substr(DATA_PREFIX.length());
        std::stringstream ss(content);
        std::string segment;
        std::vector<std::string> vals;
        while(std::getline(ss, segment, ',')) { vals.push_back(segment); }

        if (vals.size() >= 7) {
            try {
                // 先在局部变量中尝试转换
                float parsed_humidity = std::stof(vals[1]);
                
                // 物理常识过滤
                // 如果发现 ESP32 传来了不合常理的 0 值
                // 直接 return 丢弃，保留 shared_data 缓存值
                if (parsed_humidity <= 0.01f) {
                    return; 
                }

                // 若数据健康，获取互斥锁并更新全局共享状态
                std::lock_guard<std::mutex> lock(data_mtx);
                shared_data.temp = std::stof(vals[0]);
                shared_data.humidity = std::stof(vals[1]);
                shared_data.windSpeed = std::stof(vals[2]);
                shared_data.windDir = std::stoi(vals[3]);
                shared_data.pm25 = std::stoi(vals[4]);
                shared_data.pm10 = std::stoi(vals[5]);
                shared_data.timestamp = std::stoul(vals[6]);
                shared_data.isOnline = true;
            } catch (...) { /* 忽略个别转换异常 */ }
        }
    }
}

// 后台驻留的串口通信线程
void backgroundWorker() {
    global_serial = new SerialPort();
    
    while (is_running) {
        std::vector<std::string> ports = getAvailablePorts();
        bool deviceFound = false;

        for (const auto& port : ports) {
            if (global_serial->openPort(port, BAUD_RATE)) {
                global_serial->writeData(CMD_HANDSHAKE);
                auto start = std::chrono::steady_clock::now();
                std::string respBuffer = "";
                bool ack = false;

                while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(SCAN_TIMEOUT_MS)) {
                    char buf[128];
                    int n = global_serial->readData(buf, sizeof(buf)-1);
                    if (n > 0) {
                        buf[n] = 0;
                        respBuffer += buf;
                        if (respBuffer.find(ACK_HANDSHAKE) != std::string::npos) {
                            ack = true; break;
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }

                if (ack) { deviceFound = true; break; }
                else { global_serial->closePort(); }
            }
        }

        if (!deviceFound) {
            {
                std::lock_guard<std::mutex> lock(data_mtx);
                shared_data.isOnline = false;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue; 
        }

        std::string lineBuffer = "";
        auto lastPacketTime = std::chrono::steady_clock::now();

        while (is_running) {
            char chunk[128];
            int n = global_serial->readData(chunk, sizeof(chunk)-1);
            
            if (n > 0) {
                lastPacketTime = std::chrono::steady_clock::now(); 
                chunk[n] = 0;
                lineBuffer += chunk;
                
                size_t pos;
                while ((pos = lineBuffer.find('\n')) != std::string::npos) {
                    std::string line = lineBuffer.substr(0, pos);
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    parseLineToStruct(line);
                    lineBuffer.erase(0, pos + 1);
                }
            } else if (n == -1) {
                global_serial->closePort();
                break; // 硬件断开，跳出内层循环重新扫描
            } else {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPacketTime).count() > DATA_TIMEOUT_MS) {
                    std::lock_guard<std::mutex> lock(data_mtx);
                    shared_data.isOnline = false;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    if (global_serial) {
        global_serial->closePort();
        delete global_serial;
        global_serial = nullptr;
    }
}

// ==========================================
//        导出 C 接口供 Python ctypes 调用
// ==========================================
extern "C" {

    void start_monitor() {
        if (!is_running) {
            is_running = true;
            bg_thread = new std::thread(backgroundWorker);
        }
    }

    void stop_monitor() {
        if (is_running) {
            is_running = false;
            if (bg_thread && bg_thread->joinable()) {
                bg_thread->join();
                delete bg_thread;
                bg_thread = nullptr;
            }
        }
    }

    WeatherDataC get_weather_data() {
        std::lock_guard<std::mutex> lock(data_mtx);
        return shared_data;
    }

    void send_sync_cmd(uint32_t ts) {
        if (global_serial && global_serial->isConnected()) {
            std::string cmd = "CMD:SYNC_TIME:" + std::to_string(ts) + "\n";
            global_serial->writeData(cmd);
        }
    }

    void send_zero_cmd() {
        if (global_serial && global_serial->isConnected()) {
            global_serial->writeData("CMD:ZERO_WIND\n");
        }
    }

}
