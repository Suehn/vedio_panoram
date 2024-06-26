
#include "image_stitching.hpp" // 包含原始的C++代码
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(image_stitching, m) {
  py::class_<ImageStitching>(m, "ImageStitching")
      .def(py::init<>())
      .def("registration", &ImageStitching::registration)
      .def("createMask", &ImageStitching::createMask)
      .def("blending", &ImageStitching::blending);

  m.def("imread", [](const std::string &filename) {
    return cv::imread(filename, cv::IMREAD_COLOR);
  });

  m.def("imwrite", [](const std::string &filename, const cv::Mat &image) {
    return cv::imwrite(filename, image);
  });
}