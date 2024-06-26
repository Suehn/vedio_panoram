

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class ImageStitching {
public:
  ImageStitching() : ratio(0.95), min_match(10), smoothing_window_size(100) {
    orb = cv::ORB::create();
  }

  cv::Mat registration(const cv::Mat &img1, const cv::Mat &img2) {
    auto start = std::chrono::high_resolution_clock::now();

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

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(des1, des2, matches);

    auto matching_end = std::chrono::high_resolution_clock::now();
    std::cout << "Matching descriptors took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     matching_end - detect_compute_end)
                     .count()
              << " ms.\n";

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch &a, const cv::DMatch &b) {
                return a.distance < b.distance;
              });

    std::vector<cv::DMatch> good_matches;
    for (const auto &m : matches) {
      if (m.distance < ratio * matches.back().distance) {
        good_matches.push_back(m);
      }
    }

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

    cv::Mat mask3;
    cv::merge(std::vector<cv::Mat>{mask, mask, mask}, mask3);
    return mask3;
  }

  cv::Mat blending(const cv::Mat &img1, const cv::Mat &img2) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat H = registration(img1, img2);
    if (H.empty()) {
      std::cerr << "Homography could not be computed.\n";
      return cv::Mat();
    }

    int height_img1 = img1.rows;
    int width_img1 = img1.cols;
    int width_img2 = img2.cols;
    int height_panorama = height_img1;
    int width_panorama = width_img1 + width_img2;

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

    cv::Mat result = panorama1 + panorama2;

    cv::Mat result8U;
    result.convertTo(result8U, CV_8UC3);

    cv::Mat gray;
    cv::cvtColor(result8U, gray, cv::COLOR_BGR2GRAY);
    cv::Rect roi = cv::boundingRect(gray > 0);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total blending process took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms.\n";

    return result8U(roi);
  }

private:
  float ratio;
  int min_match;
  int smoothing_window_size;
  cv::Ptr<cv::ORB> orb;
};

void processImages(const std::string &img1_path, const std::string &img2_path,
                   const std::string &img3_path, const std::string &img4_path,
                   const std::string &output_path) {
  cv::Mat img1 = cv::imread(img1_path);
  cv::Mat img2 = cv::imread(img2_path);
  cv::Mat img3 = cv::imread(img3_path);
  cv::Mat img4 = cv::imread(img4_path);

  if (img1.empty() || img2.empty() || img3.empty() || img4.empty()) {
    std::cerr << "Could not open or find the images: " << img1_path << ", "
              << img2_path << ", " << img3_path << ", " << img4_path << "\n";
    return;
  }

  ImageStitching stitcher;

  // Step 1: Blend img3 and img4
  cv::Mat p3 = stitcher.blending(img3, img4);
  if (p3.empty()) {
    std::cerr << "Blending img3 and img4 failed.\n";
    return;
  }

  // Step 2: Blend img2 and p3
  cv::Mat p2 = stitcher.blending(img2, p3);
  if (p2.empty()) {
    std::cerr << "Blending img2 and p3 failed.\n";
    return;
  }

  // Step 3: Blend img1 and p2
  cv::Mat result = stitcher.blending(img1, p2);
  if (result.empty()) {
    std::cerr << "Blending img1 and p2 failed.\n";
    return;
  }

  // Save the final result
  cv::imwrite(output_path, result);
  std::cout << "Panorama image " << output_path << " has been created.\n";
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <image1> <image2> <image3> <image4> <output_image>\n";
    return -1;
  }

  processImages(argv[1], argv[2], argv[3], argv[4], argv[5]);

  return 0;
}