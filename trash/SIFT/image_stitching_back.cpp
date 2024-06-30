#include <chrono>             // 用于计时
#include <iostream>           // 用于输入输出
#include <opencv2/opencv.hpp> // OpenCV库，用于图像处理
#include <vector>             // 用于存储关键点和匹配点

// ImageStitching 类，包含图像拼接的相关方法
class ImageStitching {
public:
  // 构造函数，初始化一些参数和ORB特征检测器
  ImageStitching() : ratio(0.95), min_match(10), smoothing_window_size(400) {
    orb = cv::ORB::create();
  }

  // 图像配准函数，输入两张图像，输出变换矩阵H
  cv::Mat registration(const cv::Mat &img1, const cv::Mat &img2) {
    auto start = std::chrono::high_resolution_clock::now(); // 开始计时

    // 检测关键点并计算描述子
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, des1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, des2);

    auto detect_compute_end = std::chrono::high_resolution_clock::now();
    std::cout << "Keypoint detection and descriptor computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     detect_compute_end - start)
                     .count()
              << " ms.\n";

    // 使用暴力匹配器进行匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(des1, des2, matches);

    auto matching_end = std::chrono::high_resolution_clock::now();
    std::cout << "Matching descriptors took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     matching_end - detect_compute_end)
                     .count()
              << " ms.\n";

    // 按距离排序匹配点
    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch &a, const cv::DMatch &b) {
                return a.distance < b.distance;
              });

    // 筛选出好的匹配点
    std::vector<cv::DMatch> good_matches;
    for (const auto &m : matches) {
      if (m.distance < ratio * matches.back().distance) {
        good_matches.push_back(m);
      }
    }

    // 如果好的匹配点数目大于最小匹配数，则计算变换矩阵H
    if (good_matches.size() > min_match) {
      std::vector<cv::Point2f> image1_kp, image2_kp;
      for (const auto &m : good_matches) {
        image1_kp.push_back(kp1[m.queryIdx].pt);
        image2_kp.push_back(kp2[m.trainIdx].pt);
      }
      cv::Mat H = cv::findHomography(image2_kp, image1_kp, cv::RANSAC, 5.0);
      auto homography_end = std::chrono::high_resolution_clock::now();
      std::cout << "Homography computation took "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       homography_end - matching_end)
                       .count()
                << " ms.\n";
      return H;
    }
    return cv::Mat();
  }

  // 创建用于图像融合的掩码
  cv::Mat createMask(const cv::Mat &img1, const cv::Mat &img2,
                     const std::string &version) {
    int height_img1 = img1.rows;
    int width_img1 = img1.cols;
    int width_img2 = img2.cols;
    int height_panorama = height_img1;
    int width_panorama = width_img1 + width_img2;
    int offset = static_cast<int>(smoothing_window_size / 2);
    int barrier = width_img1 - offset;
    cv::Mat mask = cv::Mat::zeros(height_panorama, width_panorama, CV_32FC1);

    if (version == "left_image") {
      // 创建左图像的掩码
      cv::Mat left_half =
          cv::Mat::ones(height_panorama, barrier - offset, CV_32FC1);
      cv::Mat blend_zone(height_panorama, 2 * offset, CV_32FC1);
      for (int i = 0; i < 2 * offset; ++i) {
        blend_zone.col(i) = 1.0 - static_cast<float>(i) / (2 * offset);
      }
      left_half.copyTo(mask(cv::Rect(0, 0, barrier - offset, height_panorama)));
      blend_zone.copyTo(
          mask(cv::Rect(barrier - offset, 0, 2 * offset, height_panorama)));
    } else {
      // 创建右图像的掩码
      cv::Mat right_half = cv::Mat::ones(
          height_panorama, width_panorama - (barrier + offset), CV_32FC1);
      cv::Mat blend_zone(height_panorama, 2 * offset, CV_32FC1);
      for (int i = 0; i < 2 * offset; ++i) {
        blend_zone.col(i) = static_cast<float>(i) / (2 * offset);
      }
      blend_zone.copyTo(
          mask(cv::Rect(barrier - offset, 0, 2 * offset, height_panorama)));
      right_half.copyTo(
          mask(cv::Rect(barrier + offset, 0,
                        width_panorama - (barrier + offset), height_panorama)));
    }

    // 将单通道掩码转换为三通道
    cv::Mat mask3;
    cv::merge(std::vector<cv::Mat>{mask, mask, mask}, mask3);
    return mask3;
  }

  // 图像融合函数，输入两张图像，输出融合后的全景图像
  cv::Mat blending(const cv::Mat &img1, const cv::Mat &img2) {
    auto start = std::chrono::high_resolution_clock::now(); // 开始计时

    cv::Mat H = registration(img1, img2); // 获取变换矩阵H
    if (H.empty()) {
      std::cerr << "Homography could not be computed.\n";
      return cv::Mat();
    }

    int height_img1 = img1.rows;
    int width_img1 = img1.cols;
    int width_img2 = img2.cols;
    int height_panorama = height_img1;
    int width_panorama = width_img1 + width_img2;

    // 准备第一张图像的全景图
    cv::Mat panorama1 =
        cv::Mat::zeros(height_panorama, width_panorama, CV_32FC3);
    cv::Mat mask1 = createMask(img1, img2, "left_image");
    img1.convertTo(panorama1(cv::Rect(0, 0, width_img1, height_img1)),
                   CV_32FC3);
    panorama1 = panorama1.mul(mask1);

    auto panorama1_end = std::chrono::high_resolution_clock::now();
    std::cout << "First image blending took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     panorama1_end - start)
                     .count()
              << " ms.\n";

    // 准备第二张图像的全景图
    cv::Mat mask2 = createMask(img1, img2, "right_image");
    cv::Mat panorama2;
    cv::warpPerspective(img2, panorama2, H,
                        cv::Size(width_panorama, height_panorama));
    panorama2.convertTo(panorama2, CV_32FC3);
    panorama2 = panorama2.mul(mask2);

    auto panorama2_end = std::chrono::high_resolution_clock::now();
    std::cout << "Second image blending took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     panorama2_end - panorama1_end)
                     .count()
              << " ms.\n";

    // 合并两张图像
    cv::Mat result = panorama1 + panorama2;

    // 转换为8位图像
    cv::Mat result8U;
    result.convertTo(result8U, CV_8UC3);

    // 将全景图转换为灰度图，找到非零区域的最小矩形区域
    cv::Mat gray;
    cv::cvtColor(result8U, gray, cv::COLOR_BGR2GRAY);
    cv::Rect roi = cv::boundingRect(gray > 0);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total blending process took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms.\n";

    // 返回裁剪后的全景图像
    return result8U(roi);
  }

private:
  float ratio;               // 匹配比率
  int min_match;             // 最小匹配点数
  int smoothing_window_size; // 平滑窗口大小
  cv::Ptr<cv::ORB> orb;      // ORB特征检测器
};

int main() {
  // 读取两张图像
  cv::Mat img1 = cv::imread("../data/video1_frame1_d.jpg");
  cv::Mat img2 = cv::imread("../data/video2_frame1_d.jpg");

  if (img1.empty() || img2.empty()) {
    std::cerr << "Could not open or find the images!\n";
    return -1;
  }

  // 创建ImageStitching对象并进行图像拼接
  ImageStitching stitcher;
  cv::Mat result = stitcher.blending(img1, img2);

  if (!result.empty()) {
    cv::imwrite("panorama.jpg", result); // 保存全景图像
    std::cout << "Panorama image has been created.\n";
  } else {
    std::cerr << "Panorama image could not be created.\n";
  }

  return 0;
}
