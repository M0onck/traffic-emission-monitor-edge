/**
 * @file libhailo_bridge.cpp
 * @brief 绕过 Python GIL 的 Hailo NPU 元数据极速提取桥接库
 */

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

#include "hailo_objects.hpp"
#include "gst_hailo_meta.hpp"

extern "C" {
    struct DetectionBox {
        float xmin, ymin, xmax, ymax, conf;
        char label[64];
    };
}

std::vector<DetectionBox> global_detections;
std::mutex det_mutex;

// 原生 C 回调函数 (在 GStreamer 串流线程中同步执行)
static GstFlowReturn on_new_sample_c(GstAppSink *appsink, gpointer user_data) {
    // 如果流水线正在切换状态，立刻放弃本帧
    if (GST_STATE(GST_ELEMENT(appsink)) < GST_STATE_PAUSED) {
        return GST_FLOW_OK;
    }
    
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer || GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_CORRUPTED)) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    HailoROIPtr roi = get_hailo_main_roi(buffer, true);
    std::vector<DetectionBox> temp_dets;

    if (roi) {
        auto detections = roi->get_objects_typed(HAILO_DETECTION);
        
        for (const auto& obj : detections) {
            HailoDetectionPtr det = std::dynamic_pointer_cast<HailoDetection>(obj);
            if (det) {
                DetectionBox box;
                HailoBBox bbox = det->get_bbox();
                
                box.xmin = bbox.xmin(); box.ymin = bbox.ymin();
                box.xmax = bbox.xmax(); box.ymax = bbox.ymax();
                box.conf = det->get_confidence();
                
                std::string label = det->get_label();
                std::strncpy(box.label, label.c_str(), sizeof(box.label) - 1);
                box.label[sizeof(box.label) - 1] = '\0';

                temp_dets.push_back(box);
            }
        }
    }

    // 极速覆盖全局缓存
    {
        std::lock_guard<std::mutex> lock(det_mutex);
        global_detections = std::move(temp_dets);
    }

    // 瞬间放行内存，归还 NPU 缓冲池
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

extern "C" {
    // 挂载回调（Start时调用）
    void attach_hailo_sink(void* appsink_ptr) {
        if (!appsink_ptr || !G_IS_OBJECT(appsink_ptr)) return;
        GstAppSink* appsink = GST_APP_SINK(appsink_ptr);
        
        GstAppSinkCallbacks callbacks = { nullptr };
        callbacks.new_sample = on_new_sample_c;
        
        // 彻底抛弃 GObject 信号，使用最高优先级的 C 原生 Callbacks
        gst_app_sink_set_callbacks(appsink, &callbacks, nullptr, nullptr);
        std::cout << "[Hailo Bridge] SUCCESS: Native C API Callbacks Attached!" << std::endl;
    }

    // 卸载回调（Stop时调用，绝对防止销毁崩溃）
    void detach_hailo_sink(void* appsink_ptr) {
        if (!appsink_ptr || !G_IS_OBJECT(appsink_ptr)) return;
        GstAppSink* appsink = GST_APP_SINK(appsink_ptr);
        
        GstAppSinkCallbacks callbacks = { nullptr };
        // 传入空指针即卸载回调，确保后续流水线销毁时绝对安全
        gst_app_sink_set_callbacks(appsink, &callbacks, nullptr, nullptr);
        std::cout << "[Hailo Bridge] SUCCESS: Callbacks Safely Detached." << std::endl;
    }

    int get_detection_count() {
        std::lock_guard<std::mutex> lock(det_mutex);
        return global_detections.size();
    }

    void get_detections(DetectionBox* out_array, int max_size) {
        std::lock_guard<std::mutex> lock(det_mutex);
        int copy_count = std::min((int)global_detections.size(), max_size);
        for (int i = 0; i < copy_count; ++i) {
            out_array[i] = global_detections[i];
        }
    }
}
