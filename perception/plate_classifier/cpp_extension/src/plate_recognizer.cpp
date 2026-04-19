#include "plate_recognizer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#define REQUIRE_SUCCESS(expected, msg) \
    if (!expected) { \
        throw std::runtime_error(std::string(msg) + " Status: " + std::to_string(expected.status())); \
    }

// ==========================================
// 辅助结构体与数学函数
// ==========================================
struct Detection {
    float x1, y1, x2, y2;
    float conf;
    float landmarks[8];
};

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-std::clamp(x, -50.0f, 50.0f)));
}

// ==========================================
// 构造与析构 (保持上次的正确版本)
// ==========================================
HailoPlateRecognizer::HailoPlateRecognizer(const std::string& y5fu_hef, const std::string& cls_hef) 
    : y5fu_path_(y5fu_hef), cls_path_(cls_hef) {
    
    // 创建虚拟设备 (显式开启 Multi-Process Service)
    hailo_vdevice_params_t params;
    hailo_init_vdevice_params(&params); // 初始化默认参数
    params.multi_process_service = true; // 允许与其他进程 (GStreamer) 共享 NPU
    params.group_id = "SHARED";          // 指定共享组

    auto vdevice_expected = hailort::VDevice::create(params);
    REQUIRE_SUCCESS(vdevice_expected, "Failed to create VDevice (MPS 模式)");
    vdevice_ = vdevice_expected.release();

    uint32_t timeout_ms = 10000;
    uint32_t queue_size = 32;
    std::string network_name = "";

    // 1. 加载 Detector (Y5FU)
    auto det_hef_expected = hailort::Hef::create(y5fu_path_);
    REQUIRE_SUCCESS(det_hef_expected, "Failed to parse Detector HEF");
    auto det_hef = det_hef_expected.release();

    auto configure_params_det = vdevice_->create_configure_params(det_hef);
    REQUIRE_SUCCESS(configure_params_det, "Failed to create Det configure params");
    auto det_network_groups = vdevice_->configure(det_hef, configure_params_det.value());
    REQUIRE_SUCCESS(det_network_groups, "Failed to configure Detector");
    det_network_group_ = det_network_groups.value()[0];

    auto det_in_params_expected = det_network_group_->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, timeout_ms, queue_size, network_name);
    det_inputs_ = hailort::VStreamsBuilder::create_input_vstreams(*det_network_group_, det_in_params_expected.value()).release();

    auto det_out_params_expected = det_network_group_->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, timeout_ms, queue_size, network_name);
    det_outputs_ = hailort::VStreamsBuilder::create_output_vstreams(*det_network_group_, det_out_params_expected.value()).release();

    // 2. 加载 Classifier (LiteModel)
    auto classifier_hef = hailort::Hef::create(cls_path_).release();
    cls_network_group_ = vdevice_->configure(classifier_hef, vdevice_->create_configure_params(classifier_hef).value()).value()[0];
    
    auto cls_in_params = cls_network_group_->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, timeout_ms, queue_size, network_name).value();
    cls_inputs_ = hailort::VStreamsBuilder::create_input_vstreams(*cls_network_group_, cls_in_params).release();

    auto cls_out_params = cls_network_group_->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, timeout_ms, queue_size, network_name).value();
    cls_outputs_ = hailort::VStreamsBuilder::create_output_vstreams(*cls_network_group_, cls_out_params).release();
}

HailoPlateRecognizer::~HailoPlateRecognizer() {}

// ==========================================
// 核心处理函数 (完全接管 Python Pipeline)
// ==========================================
PlateResult HailoPlateRecognizer::process(py::array_t<uint8_t> input_image) {
    PlateResult result;
    result.color_type = "unknown";
    result.confidence = 0.0f;

    // 此时仍然持有 GIL，安全地调用 Python C API 获取底层指针
    py::buffer_info buf = input_image.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Input image must have 3 dimensions (H, W, C)");
    }
    cv::Mat cv_img(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
    if (cv_img.empty()) return result;

    // 图像指针已经拿到，开始极其耗时的 NPU 推理和 OpenCV 计算
    // 此时手动实例化 gil_scoped_release，它会在作用域内自动释放 GIL 给 Python 主线程
    // 当 process 函数运行结束时，它会自动重新获取 GIL 并将 C++ 结构体安全转回 Python
    py::gil_scoped_release release;

    // ---------------------------------------------------------
    // 第一步：Y5FU 预处理 (Letterbox)
    // ---------------------------------------------------------
    // 从 VStream 获取模型需要的输入尺寸 (通常是 640x640)
    int model_h = det_inputs_[0].get_info().shape.height;
    int model_w = det_inputs_[0].get_info().shape.width;
    
    float ratio = std::min((float)model_w / cv_img.cols, (float)model_h / cv_img.rows);
    int new_w = std::round(cv_img.cols * ratio);
    int new_h = std::round(cv_img.rows * ratio);
    int pad_w = (model_w - new_w) / 2;
    int pad_h = (model_h - new_h) / 2;

    cv::Mat resized, padded;
    cv::resize(cv_img, resized, cv::Size(new_w, new_h));
    cv::copyMakeBorder(resized, padded, pad_h, model_h - new_h - pad_h, pad_w, model_w - new_w - pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

    // ---------------------------------------------------------
    // 第二步：Y5FU 推理
    // ---------------------------------------------------------
    auto write_status = det_inputs_[0].write(hailort::MemoryView(padded.data, padded.total() * padded.elemSize()));
    if (write_status != HAILO_SUCCESS) return result;

    // 读取所有的输出特征图
    std::vector<std::vector<float>> raw_outputs(det_outputs_.size());
    for (size_t i = 0; i < det_outputs_.size(); ++i) {
        size_t frame_size = det_outputs_[i].get_frame_size();
        raw_outputs[i].resize(frame_size / sizeof(float));
        det_outputs_[i].read(hailort::MemoryView(raw_outputs[i].data(), frame_size));
    }

    // ---------------------------------------------------------
    // 第三步：后处理解码 (翻译自 _decode_raw_logits)
    // ---------------------------------------------------------
    float strides[] = {8.0f, 16.0f, 32.0f};
    float anchors[3][3][2] = {
        {{4, 5}, {8, 10}, {13, 16}},
        {{23, 29}, {43, 55}, {73, 105}},
        {{146, 217}, {231, 300}, {335, 433}}
    };

    std::vector<Detection> valid_dets;
    float conf_thresh = 0.5f; // 来自 multitask_detect.py

    for (size_t i = 0; i < det_outputs_.size(); ++i) {
        int grid_h = det_outputs_[i].get_info().shape.height;
        int grid_w = det_outputs_[i].get_info().shape.width;
        
        // 自动匹配当前输出属于哪个 Stride (通过分辨率计算)
        int stride_idx = -1;
        if (model_w / grid_w == 8) stride_idx = 0;
        else if (model_w / grid_w == 16) stride_idx = 1;
        else if (model_w / grid_w == 32) stride_idx = 2;
        if (stride_idx == -1) continue;

        float stride = strides[stride_idx];
        float* data_ptr = raw_outputs[i].data();

        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                for (int a = 0; a < 3; ++a) { // 3个anchor
                    // 15 个通道的偏移量 (由于内存排布可能是 HWC，此处简化，请根据实际排布微调)
                    int idx = (y * grid_w * 3 + x * 3 + a) * 15;
                    
                    float obj_conf = sigmoid(data_ptr[idx + 4]);
                    if (obj_conf < conf_thresh) continue;

                    Detection det;
                    det.conf = obj_conf;

                    // BBox 解码
                    float bx = (sigmoid(data_ptr[idx + 0]) * 2.0f - 0.5f + x) * stride;
                    float by = (sigmoid(data_ptr[idx + 1]) * 2.0f - 0.5f + y) * stride;
                    float bw = std::pow(sigmoid(data_ptr[idx + 2]) * 2.0f, 2.0f) * anchors[stride_idx][a][0];
                    float bh = std::pow(sigmoid(data_ptr[idx + 3]) * 2.0f, 2.0f) * anchors[stride_idx][a][1];

                    // 映射回原图尺寸并扣除 Letterbox padding
                    det.x1 = ((bx - bw / 2.0f) - pad_w) / ratio;
                    det.y1 = ((by - bh / 2.0f) - pad_h) / ratio;
                    det.x2 = ((bx + bw / 2.0f) - pad_w) / ratio;
                    det.y2 = ((by + bh / 2.0f) - pad_h) / ratio;

                    // 关键点解码
                    for (int k = 0; k < 4; ++k) {
                        float lmk_x = (sigmoid(data_ptr[idx + 5 + k*2]) * 2.0f - 0.5f + x) * stride;
                        float lmk_y = (sigmoid(data_ptr[idx + 5 + k*2 + 1]) * 2.0f - 0.5f + y) * stride;
                        det.landmarks[k*2] = (lmk_x - pad_w) / ratio;
                        det.landmarks[k*2+1] = (lmk_y - pad_h) / ratio;
                    }
                    valid_dets.push_back(det);
                }
            }
        }
    }

    if (valid_dets.empty()) return result;

    // 简单寻找最高置信度的车牌 (等效于 pipeline.py 中的 best_idx 逻辑)
    auto best_det = std::max_element(valid_dets.begin(), valid_dets.end(),
        [](const Detection& a, const Detection& b) { return a.conf < b.conf; });

    // 填充 Python 层需要的相对归一化关键点
    for (int k = 0; k < 8; ++k) {
        if (k % 2 == 0) result.landmarks.push_back(best_det->landmarks[k] / cv_img.cols);
        else result.landmarks.push_back(best_det->landmarks[k] / cv_img.rows);
    }

    // ---------------------------------------------------------
    // 第四步：透视变换 (翻译自 get_rotate_crop_image)
    // ---------------------------------------------------------
    cv::Point2f src_pts[4];
    for (int i = 0; i < 4; ++i) {
        src_pts[i] = cv::Point2f(best_det->landmarks[i*2], best_det->landmarks[i*2+1]);
    }

    // 动态计算透视后的宽高
    float width_top = cv::norm(src_pts[0] - src_pts[1]);
    float width_bottom = cv::norm(src_pts[2] - src_pts[3]);
    float height_left = cv::norm(src_pts[0] - src_pts[3]);
    float height_right = cv::norm(src_pts[1] - src_pts[2]);
    int crop_w = static_cast<int>(std::max(width_top, width_bottom));
    int crop_h = static_cast<int>(std::max(height_left, height_right));

    cv::Point2f dst_pts[4] = {
        cv::Point2f(0, 0), cv::Point2f(crop_w, 0),
        cv::Point2f(crop_w, crop_h), cv::Point2f(0, crop_h)
    };

    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(cv_img, warped, M, cv::Size(crop_w, crop_h), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    
    if (crop_h * 1.0 / crop_w >= 1.5) {
        cv::rotate(warped, warped, cv::ROTATE_90_CLOCKWISE);
    }

    // ---------------------------------------------------------
    // 第五步：LiteModel 颜色分类
    // ---------------------------------------------------------
    int cls_h = cls_inputs_[0].get_info().shape.height;
    int cls_w = cls_inputs_[0].get_info().shape.width;

    cv::Mat cls_resized;
    cv::resize(warped, cls_resized, cv::Size(cls_w, cls_h));
    cv::cvtColor(cls_resized, cls_resized, cv::COLOR_BGR2RGB);

    auto cls_write = cls_inputs_[0].write(hailort::MemoryView(cls_resized.data, cls_resized.total() * cls_resized.elemSize()));
    if (cls_write != HAILO_SUCCESS) return result;

    size_t cls_out_size = cls_outputs_[0].get_frame_size();
    std::vector<float> cls_raw(cls_out_size / sizeof(float));
    cls_outputs_[0].read(hailort::MemoryView(cls_raw.data(), cls_out_size));

    // 寻找最大概率颜色 (按 pipeline.py 中的 PLATE_TYPE 映射)
    int best_color_idx = 0;
    float max_prob = -1.0f;
    for (size_t i = 0; i < cls_raw.size(); ++i) {
        if (cls_raw[i] > max_prob) {
            max_prob = cls_raw[i];
            best_color_idx = i;
        }
    }

    result.confidence = max_prob;
    // 参考 pipeline.py 的 idx 映射
    if (best_color_idx == 0) result.color_type = "yellow";      // PLATE_TYPE_YELLOW
    else if (best_color_idx == 1) result.color_type = "blue";   // PLATE_TYPE_BLUE
    else if (best_color_idx == 2) result.color_type = "green";  // PLATE_TYPE_GREEN
    else result.color_type = "green"; // 兜底

    return result;
}
