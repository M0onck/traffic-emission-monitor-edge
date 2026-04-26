/**
 * @file transmitter.cpp
 * @brief ESP32 气象站发送端固件 (双向通信/数据精简版)
 */

#include <Arduino.h>
#include <HardwareSerial.h>
#include <esp_now.h>
#include <WiFi.h>

#define RX_PIN 17
#define TX_PIN 18
#define LED_PIN 2
#define BAUD_RATE 4800
#define SENSOR_ADDR 0x01
#define START_REG 0x01F4      // 500
#define REG_COUNT 9           // 只需要读取 500 到 508 共9个寄存器
#define SAMPLING_INTERVAL 1000 // 1Hz 刷新率

HardwareSerial MySerial(1);
uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// 裁剪后的精简数据包
typedef struct struct_message {
  float temp;
  float humidity;
  float windSpeed;
  int windDir;
  int pm25;
  int pm10;
  uint32_t timestamp; // 同步后的绝对时间戳
} struct_message;

// 控制指令包
typedef struct cmd_packet {
  uint8_t cmdType;    // 1: 时钟同步, 2: 风速调零
  uint32_t timestamp; // 仅在 cmdType==1 时有效
} cmd_packet;

struct_message myData;
esp_now_peer_info_t peerInfo;

// 内部时间偏移量 = 上位机真实时间 - 运行时(millis/1000)
int32_t timeOffsetSeconds = 0;
volatile bool flag_pending_zero = false; // 调零指令待处理标志

uint16_t calculateCRC(const byte* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int j = 0; j < 8; j++) {
      if (crc & 0x0001) crc = (crc >> 1) ^ 0xA001;
      else crc >>= 1;
    }
  }
  return crc;
}

// 执行风速调零
void executeWindZero() {
  // 1. 发送前清空 RX 缓冲
  while(MySerial.available()) MySerial.read();
  
  // 风速调零用 Modbus 指令
  byte cmd[] = {0x01, 0xAB, 0x01, 0x06, 0x01, 0x02, 0x00, 0x5A, 0xA9, 0xCD, 0xEA, 0xFE};
  
  // 调试打印
  Serial.print("SYS: [TX] 实际发往 RS485 的调零指令(HEX): ");
  int sendLength = 12; // 如果厂家协议是 12 字节，这里改为 12，并注销上面计算 CRC 的代码
  for(int i = 0; i < sendLength; i++) {
    Serial.printf("%02X ", cmd[i]);
  }
  Serial.println();

  // 2. 发送并确保物理层输出完毕
  MySerial.write(cmd, sendLength);
  MySerial.flush(); // 阻塞直到 TX 缓冲区的数据全被推上 485 总线
  
  // 3. 监听气象站的真实回应 (设置 1.5 秒超时窗口)
  Serial.print("SYS: [RX] 等待气象站响应(HEX): ");
  unsigned long startT = millis();
  bool hasResponse = false;
  
  while(millis() - startT < 1500) { 
    if (MySerial.available()) {
      hasResponse = true;
      byte b = MySerial.read();
      Serial.printf("%02X ", b);
    }
  }
  
  if (!hasResponse) {
    Serial.println("[无响应 - 设备可能未收到或忽略了指令]");
  } else {
    Serial.println(); // 换行
  }
}

// 接收 ESP-NOW 指令回调
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  if (len == sizeof(cmd_packet)) {
    cmd_packet cmd;
    memcpy(&cmd, incomingData, sizeof(cmd));
    
    if (cmd.cmdType == 1) { // 时钟同步
      timeOffsetSeconds = cmd.timestamp - (millis() / 1000);
      Serial.printf("时钟已同步, 偏移量: %d\n", timeOffsetSeconds);
    } 
    else if (cmd.cmdType == 2) { // 风速调零
      flag_pending_zero = true; 
      Serial.println("SYS: 收到调零请求，已加入主循环队列");
    }
  }
}

void setup() {
  Serial.begin(115200);
  MySerial.begin(BAUD_RATE, SERIAL_8N1, RX_PIN, TX_PIN);
  pinMode(LED_PIN, OUTPUT);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) return;

  // 注册接收与发送回调
  esp_now_register_recv_cb(OnDataRecv);
  
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);
}

void loop() {
  // 如果存在挂起的调零任务，优先处理
  if (flag_pending_zero) {
    flag_pending_zero = false; // 清除标志
    executeWindZero();         // 由 loop 任务亲自执行
    return;                    // 强制结束当前 loop 周期，防止后续的 03 读取指令发出
  }

  // 安全起见，每次发起新轮询前彻底清空 RX 缓冲区
  while (MySerial.available()) {
    MySerial.read();
  }

  // 1. 发送 Modbus 读指令
  byte cmd[] = {
    SENSOR_ADDR, 0x03, 
    (byte)(START_REG >> 8), (byte)(START_REG & 0xFF), 
    (byte)(REG_COUNT >> 8), (byte)(REG_COUNT & 0xFF), 
    0, 0 
  };
  uint16_t crc = calculateCRC(cmd, 6);
  cmd[6] = crc & 0xFF;  cmd[7] = crc >> 8;
  MySerial.write(cmd, 8);
  digitalWrite(LED_PIN, HIGH);
  
  // 2. 接收响应 (1(址)+1(功)+1(长)+18(数据)+2(CRC) = 23字节)
  byte buf[32]; 
  int len = 0;
  unsigned long startT = millis();
  while (millis() - startT < 500) { 
    if (MySerial.available()) {
      buf[len++] = MySerial.read();
      if (len >= 23) break; 
    }
  }
  digitalWrite(LED_PIN, LOW);

  // 3. 解析并广播
  if (len >= 23) {
    auto getVal = [&](int index) -> uint16_t {
      int offset = 3 + index * 2; 
      return (buf[offset] << 8) | buf[offset + 1];
    };

    myData.windSpeed = getVal(0) * 0.1;
    myData.windDir   = getVal(3);
    myData.humidity  = getVal(4) * 0.1;
    myData.temp      = (int16_t)getVal(5) * 0.1;
    myData.pm25      = getVal(7);
    myData.pm10      = getVal(8);
    // 附加上同步后的时间戳 (若未同步则相当于运行秒数)
    myData.timestamp = (millis() / 1000) + timeOffsetSeconds; 

    esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));
  }
  delay(SAMPLING_INTERVAL); 
}
