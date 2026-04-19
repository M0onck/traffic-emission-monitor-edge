#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 必须包含，用于自动转换 std::string 和 std::vector
#include "plate_recognizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(hailo_ocr_cpp, m) {
    m.doc() = "Hailo-8 C++ Native OCR Extension for Traffic Monitor";

    // 绑定结果结构体，使其在 Python 中可以作为对象访问属性
    py::class_<PlateResult>(m, "PlateResult")
        .def_readonly("color_type", &PlateResult::color_type)
        .def_readonly("confidence", &PlateResult::confidence)
        .def_readonly("landmarks", &PlateResult::landmarks);

    // 绑定核心识别类
    py::class_<HailoPlateRecognizer>(m, "HailoPlateRecognizer")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("y5fu_hef_path"), py::arg("cls_hef_path"))
        
        .def("process", &HailoPlateRecognizer::process, py::arg("input_image"));
}
