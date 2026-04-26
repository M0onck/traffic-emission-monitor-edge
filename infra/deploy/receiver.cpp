/**
 * @file receiver.cpp
 * @brief ESP32 接收端固件 (双向通信网关)
 */

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>

const String CMD_HANDSHAKE = "CMD:WHO_ARE_YOU";
const String ACK_HANDSHAKE = "ACK:WEATHER_RX_V1";

typedef struct struct_message {
  float temp;
  float humidity;
  float windSpeed;
  int windDir;
  int pm25;
  int pm10;
  uint32_t timestamp;
} struct_message;

typedef struct cmd_packet {
  uint8_t cmdType; 
  uint32_t timestamp; 
} cmd_packet;

struct_message myData;
uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
esp_now_peer_info_t peerInfo;

void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  if (len == sizeof(myData)) {
    memcpy(&myData, incomingData, sizeof(myData));
    // 仅输出 6 个要素和时间戳
    Serial.printf("DATA:%.2f,%.2f,%.2f,%d,%d,%d,%u\n", 
                  myData.temp, 
                  myData.humidity, 
                  myData.windSpeed, 
                  myData.windDir,
                  myData.pm25, 
                  myData.pm10,
                  myData.timestamp
                );
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  
  if (esp_now_init() != ESP_OK) return;
  esp_now_register_recv_cb(OnDataRecv);

  // 注册广播地址用于发送指令
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == CMD_HANDSHAKE) {
      Serial.println(ACK_HANDSHAKE);
    } 
    else if (input.startsWith("CMD:SYNC_TIME:")) {
      String tsStr = input.substring(14);
      cmd_packet cmd = {1, (uint32_t)tsStr.toInt()};
      esp_now_send(broadcastAddress, (uint8_t*)&cmd, sizeof(cmd));
      Serial.println("SYS:TIME_SYNC_FORWARDED");
    }
    else if (input == "CMD:ZERO_WIND") {
      cmd_packet cmd = {2, 0};
      esp_now_send(broadcastAddress, (uint8_t*)&cmd, sizeof(cmd));
      Serial.println("SYS:ZERO_WIND_FORWARDED");
    }
  }
}
