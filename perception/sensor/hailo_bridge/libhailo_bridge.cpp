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

static GstFlowReturn on_new_sample_c(GstElement *appsink, gpointer user_data) {
    // 增加内部静态心跳计数器
    static int frame_count = 0;
    frame_count++;

    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    HailoROIPtr roi = get_hailo_main_roi(buffer, true);
    std::vector<DetectionBox> temp_dets;

    if (roi) {
        auto detections = roi->get_objects_typed(HAILO_DETECTION);
        
        // 每隔 30 帧 (约 1-3 秒) 在终端打印一次底层心跳
        if (frame_count % 30 == 0) {
            std::cout << "[Hailo Bridge] Frame " << frame_count 
                      << " | Detected Cars: " << detections.size() << std::endl;
        }

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
    } else {
        if (frame_count % 30 == 0) {
            std::cout << "[Hailo Bridge] Frame " << frame_count << " | WARNING: NO ROI FOUND!" << std::endl;
        }
    }

    {
        std::lock_guard<std::mutex> lock(det_mutex);
        global_detections = std::move(temp_dets);
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

extern "C" {
    void attach_hailo_sink(void* appsink_ptr) {
        if (!appsink_ptr || !G_IS_OBJECT(appsink_ptr)) {
            std::cerr << "[Hailo Bridge] FATAL: Python passed an invalid memory pointer!" << std::endl;
            return;
        }
        
        GstElement* appsink = GST_ELEMENT(appsink_ptr);
        g_object_set(G_OBJECT(appsink), "emit-signals", TRUE, "sync", FALSE, NULL);
        g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample_c), NULL);
        
        std::cout << "[Hailo Bridge] SUCCESS: C++ Callbacks Attached! GIL is now bypassed." << std::endl;
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
