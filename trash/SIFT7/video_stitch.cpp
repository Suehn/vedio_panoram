
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

class ImageStitching {
public:
  ImageStitching() : ratio(0.95), min_match(10), smoothing_window_size(100) {
    orb = cv::ORB::create();
    prev_H = cv::Mat::eye(3, 3, CV_64F);
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
      std::cout << "Homography Matrix: \n" << H << "\n";
      return H;
    }
    return cv::Mat::eye(3, 3,
                        CV_64F); // Return identity matrix if not enough matches
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

  cv::Mat blending(const cv::Mat &img1, const cv::Mat &img2, int frame_count) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat H;

    if (frame_count % 1 == 0 && frame_count != 0) {
      cv::Mat new_H; // 在外部声明 new_H
      if (frame_count == 1) {
        new_H = prev_H; // 直接使用 new_H
      } else {
        new_H = registration(img1, img2); // 直接使用 new_H
      }
      H = 0.002 * new_H + 0.9998 * prev_H;
      prev_H = H;
    } else if (frame_count == 0) {
      H = registration(img1, img2);
    } else {
      H = prev_H;
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
  cv::Mat prev_H;
  cv::Ptr<cv::ORB> orb;
};

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <video1> <video2> <output_folder>\n";
    return -1;
  }

  std::string video1_path = argv[1];
  std::string video2_path = argv[2];
  std::string output_folder = argv[3];

  cv::VideoCapture cap1(video1_path);
  cv::VideoCapture cap2(video2_path);

  if (!cap1.isOpened() || !cap2.isOpened()) {
    std::cerr << "Could not open one of the videos.\n";
    return -1;
  }

  ImageStitching stitcher;

  int frame_count = 0;
  cv::Mat frame1, frame2, result;
  while (true) {
    cap1 >> frame1;
    cap2 >> frame2;

    if (frame1.empty() || frame2.empty()) {
      break;
    }

    result = stitcher.blending(frame1, frame2, frame_count);

    if (result.empty()) {
      std::cerr << "Stitching failed for frame: " << frame_count << "\n";
      continue;
    }

    std::string output_path =
        output_folder + "/frame_" + std::to_string(frame_count) + ".jpg";
    cv::imwrite(output_path, result);
    std::cout << "Stitched frame " << frame_count << " saved as " << output_path
              << ".\n";

    frame_count++;
  }

  cap1.release();
  cap2.release();

  return 0;
}