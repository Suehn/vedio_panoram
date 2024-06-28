import cv2 as cv
import numpy as np
from typing import Tuple, List, Optional, Dict
import numpy as np

def read_frames_from_videos(video_path1: str, video_path2: str, frame_number: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从两个视频文件中提取特定帧并返回。

    参数:
    video_path1 (str): 第一个视频文件的路径。
    video_path2 (str): 第二个视频文件的路径。
    frame_number (int): 要提取的帧号。

    返回:
    Tuple[np.ndarray, np.ndarray]: 从两个视频文件中提取的帧。
    """
    def read_frame(video_path: str, frame_number: int) -> np.ndarray:
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"无法从 {video_path} 提取帧 {frame_number}")
        cap.release()
        return frame
    
    frame1 = read_frame(video_path1, frame_number)
    frame2 = read_frame(video_path2, frame_number)
    
    return frame1, frame2


def resize_to_equal_width(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将两个图像缩放为相同的宽度，宽度取较小图像的宽度。

    参数:
    img1 (np.ndarray): 第一个图像。
    img2 (np.ndarray): 第二个图像。

    返回:
    Tuple[np.ndarray, np.ndarray]: 缩放后的两个图像。
    """
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    if width1 < width2:
        new_width = width1
        new_height2 = int((height2 / width2) * new_width)
        resized_img2 = cv.resize(img2, (new_width, new_height2), interpolation=cv.INTER_AREA)
        resized_img1 = img1
    else:
        new_width = width2
        new_height1 = int((height1 / width1) * new_width)
        resized_img1 = cv.resize(img1, (new_width, new_height1), interpolation=cv.INTER_AREA)
        resized_img2 = img2

    return resized_img1, resized_img2


def undistort_and_rotate(image: np.ndarray, dist_coeffs: np.ndarray = np.array([-0.0733, 0.0833, 0, 0]), 
                         camera_matrix: np.ndarray = np.array([[800, 0, 640], 
                                                               [0, 800, 360], 
                                                               [0, 0, 1]], dtype=np.float32), 
                         angle: float = 0) -> np.ndarray:
    """
    对图像进行畸变矫正和旋转。

    参数:
    image (np.ndarray): 输入图像。
    dist_coeffs (np.ndarray): 畸变系数。
    camera_matrix (np.ndarray): 相机矩阵，默认值为常用参数。
    angle (float): 旋转角度，默认为0。

    返回:
    np.ndarray: 矫正和旋转后的图像。
    """
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (image.shape[1], image.shape[0]), 1, (image.shape[1], image.shape[0]))
    undistorted_image = cv.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    (h, w) = undistorted_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv.warpAffine(undistorted_image, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated_image


def detect_and_compute_features(img: np.ndarray, method: str = 'SIFT', x_range: List[float] = [0.0, 1.0]) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    """
    使用ORB或SIFT算法检测图像特征点并计算描述子，同时只保留指定横坐标范围内的特征点和描述子。

    参数:
    img (np.ndarray): 输入图像。
    method (str): 使用的特征检测算法，'ORB'或'SIFT'。
    x_range (List[float]): 要保留的横坐标范围，取值范围在[0, 1]之间，默认为[0.0, 1.0]表示保留整个图像的特征点。

    返回:
    Tuple[List[cv.KeyPoint], np.ndarray]: 特征点列表和描述子矩阵。
    """
    if method == 'ORB':
        detector = cv.ORB_create()
    elif method == 'SIFT':
        detector = cv.SIFT_create()
    else:
        raise ValueError("method 参数必须是 'ORB' 或 'SIFT'")

    keypoints, descriptors = detector.detectAndCompute(img, None)

    if not keypoints or descriptors is None:
        return [], np.array([])

    img_width = img.shape[1]
    x_min = x_range[0] * img_width
    x_max = x_range[1] * img_width

    filtered_keypoints = []
    filtered_descriptors = []

    for kp, desc in zip(keypoints, descriptors):
        if x_min <= kp.pt[0] <= x_max:
            filtered_keypoints.append(kp)
            filtered_descriptors.append(desc)

    filtered_descriptors = np.array(filtered_descriptors)

    return filtered_keypoints, filtered_descriptors



def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray) -> List[cv.DMatch]:
    """
    使用FLANN算法匹配两个图像的特征描述子。

    参数:
    descriptors1 (np.ndarray): 第一个图像的描述子。
    descriptors2 (np.ndarray): 第二个图像的描述子。

    返回:
    List[cv.DMatch]: 优质匹配点对。
    """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.65 * n.distance]

    return good_matches

def compute_homography(kp1: List[cv.KeyPoint], kp2: List[cv.KeyPoint], matches: List[cv.DMatch], min_match_count: int = 10) -> Optional[np.ndarray]:

    """
    从匹配点计算单应性矩阵。

    参数:
    kp1 (List[cv.KeyPoint]): 第一张图像的特征点。
    kp2 (List[cv.KeyPoint]): 第二张图像的特征点。
    matches (List[cv.DMatch]): 好的匹配点。
    min_match_count (int): 最少匹配点数，默认值为10。

    返回:
    Optional[np.ndarray]: 计算得到的单应性矩阵，如果匹配点不足则返回None。
    """
    if len(matches) >= min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        return M
    else:
        print("匹配点不足！")
        return None

def stitch_images_with_blending(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    使用单应性矩阵拼接两张图像，并在重叠区域使用加权平均方法进行混合。

    参数:
    img1 (np.ndarray): 第一张图像。
    img2 (np.ndarray): 第二张图像。
    H (np.ndarray): 单应性矩阵。

    返回:
    np.ndarray: 拼接后的图像。
    """
    # 获取图像尺寸
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # 使用单应性矩阵将 img2 变换到 img1 的平面
    warp_img2 = cv.warpPerspective(img2, np.linalg.inv(H), (width1 + width2, height1))

    # 在变换后的图像上复制 img1
    result = warp_img2.copy()
    result[0:height1, 0:width1] = img1

    # 找到重叠区域
    rows, cols = img1.shape[:2]
    left, right = 0, cols
    
    for col in range(0, cols):
        if img1[:, col].any() and warp_img2[:, col].any():
            left = col
            break
    
    for col in range(cols - 1, 0, -1):
        if img1[:, col].any() and warp_img2[:, col].any():
            right = col
            break
    
    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            if not img1[row, col].any():
                res[row, col] = warp_img2[row, col]
            elif not warp_img2[row, col].any():
                res[row, col] = img1[row, col]
            else:
                srcimg_len = float(abs(col - left))
                warpimg_len = float(abs(col - right))
                alpha = srcimg_len / (srcimg_len + warpimg_len)
                res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warp_img2[row, col] * alpha, 0, 255)
    
    warp_img2[0:img1.shape[0], 0:img1.shape[1]] = res
    return warp_img2
