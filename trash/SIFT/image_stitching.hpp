
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class ImageStitching {
public:
  ImageStitching();
  cv::Mat registration(const cv::Mat &img1, const cv::Mat &img2);
  cv::Mat createMask(const cv::Mat &img1, const cv::Mat &img2,
                     const std::string &version);
  cv::Mat blending(const cv::Mat &img1, const cv::Mat &img2);

private:
  float ratio;
  int min_match;
  int smoothing_window_size;
  cv::Ptr<cv::ORB> orb;
};