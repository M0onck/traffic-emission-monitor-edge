/**
 * @file libhailo_bridge.cpp
 * @brief 绕过 Python GIL 的 Hailo NPU 元数据极速提取桥接库
 * @details 运行在底层 C++ 线程，提取 BBox 后将数据缓存在全局 Mutex 结构中
 */

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

// Hailo TAPPAS 核心 API 头文件 (适用最新版 Tappas)
#include "hailo_objects.hpp"
#include "gst_hailo_meta.hpp"

// ==========================================
// 1. 数据结构定义 (必须与 Python ctypes 保持精确对齐)
// ==========================================
extern "C" {
    struct DetectionBox {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        float conf;
        char label[64];
    };
}

// 全局缓存与锁
std::vector<DetectionBox> global_detections;
std::mutex det_mutex;

// ==========================================
// 2. 原生 AppSink 异步回调 (运行在 GStreamer 线程，绝对无锁)
// ==========================================
static GstFlowReturn on_new_sample_c(GstElement *appsink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    // 【核心】提取 Hailo 区域对象 (ROI)
    HailoROIPtr roi = get_hailo_main_roi(buffer, true);
    std::vector<DetectionBox> temp_dets;

    if (roi) {
        // 过滤并提取所有检测类型 (HAILO_DETECTION)
        for (const auto& obj : roi->get_objects_typed(HAILO_DETECTION)) {
            HailoDetectionPtr det = std::dynamic_pointer_cast<HailoDetection>(obj);
            if (det) {
                DetectionBox box;
                HailoBBox bbox = det->get_bbox();
                
                box.xmin = bbox.xmin();
                box.ymin = bbox.ymin();
                box.xmax = bbox.xmax();
                box.ymax = bbox.ymax();
                box.conf = det->get_confidence();
                
                // 安全拷贝字符串
                std::string label = det->get_label();
                std::strncpy(box.label, label.c_str(), sizeof(box.label) - 1);
                box.label[sizeof(box.label) - 1] = '\0';

                temp_dets.push_back(box);
            }
        }
    }

    // 【极速交换】锁住非常短的时间，仅做内存移动
    {
        std::lock_guard<std::mutex> lock(det_mutex);
        global_detections = std::move(temp_dets);
    }

    // 立刻释放 Sample，瞬间归还 NPU 底层 RequestWrap
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// ==========================================
// 3. 导出 C 接口供 Python 调用
// ==========================================
extern "C" {

    // 绑定底层的 AppSink 信号，此后数据由 C++ 自动捕获
    void attach_hailo_sink(void* appsink_ptr) {
        GstElement* appsink = GST_ELEMENT(appsink_ptr);
        
        // 激活信号发射，关闭同步死锁
        g_object_set(G_OBJECT(appsink), "emit-signals", TRUE, "sync", FALSE, NULL);
        
        // 将 new-sample 信号挂载到原生 C++ 函数
        g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample_c), NULL);
        
        std::cout << "[Hailo Bridge] Attached to AppSink! Python GIL bypassed." << std::endl;
    }

    // 获取当前帧的检测框数量
    int get_detection_count() {
        std::lock_guard<std::mutex> lock(det_mutex);
        return global_detections.size();
    }

    // 拷贝数据到 Python 预分配的数组
    void get_detections(DetectionBox* out_array, int max_size) {
        std::lock_guard<std::mutex> lock(det_mutex);
        int copy_count = std::min((int)global_detections.size(), max_size);
        for (int i = 0; i < copy_count; ++i) {
            out_array[i] = global_detections[i];
        }
    }
}
