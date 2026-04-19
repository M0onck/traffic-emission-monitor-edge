#pragma once
#include <string>
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <hailo/hailort.hpp> // 引入 HailoRT C++ 头文件

namespace py = pybind11;

struct PlateResult {
    std::string color_type;
    float confidence;
    std::vector<float> landmarks;
};

class HailoPlateRecognizer {
public:
    HailoPlateRecognizer(const std::string& y5fu_hef_path, const std::string& cls_hef_path);
    ~HailoPlateRecognizer();

    PlateResult process(py::array_t<uint8_t> input_image);

private:
    std::string y5fu_path_;
    std::string cls_path_;

    // --- Hailo 硬件句柄 ---
    std::unique_ptr<hailort::VDevice> vdevice_;
    
    // 网络配置组
    std::shared_ptr<hailort::ConfiguredNetworkGroup> det_network_group_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> cls_network_group_;
    
    // Y5FU 探测器的输入输出流
    std::vector<hailort::InputVStream> det_inputs_;
    std::vector<hailort::OutputVStream> det_outputs_;
    
    // LiteModel 分类器的输入输出流
    std::vector<hailort::InputVStream> cls_inputs_;
    std::vector<hailort::OutputVStream> cls_outputs_;
};
