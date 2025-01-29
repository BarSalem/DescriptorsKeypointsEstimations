import time

import cv2
import numpy as np


def calculate_function_and_duration(func, *args, **kwargs):
    start_time = time.time()
    repeatability, localization_error = func(*args, **kwargs)
    duration = time.time() - start_time
    return {
        "repeatability": repeatability,
        "localization_error": localization_error,
        "duration": duration
    }


def calculate_descriptor_function_and_duration(func, *args, **kwargs):
    start_time = time.time()
    precision = func(*args, **kwargs)
    duration = time.time() - start_time
    return {
        "precision": precision,
        "duration": duration
    }


def apply_rotation(keypoints, img_shape, angle):
    """
    Rotate keypoints around the image center by a given angle.
    """
    h, w = img_shape
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Convert keypoints to homogeneous coordinates for transformation
    points = np.array([kp.pt for kp in keypoints])
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    # Rotate points
    rotated_points = (rot_mat @ points_h.T).T
    rotated_keypoints = [cv2.KeyPoint(x, y, 1) for x, y in rotated_points]
    return rotated_keypoints


def apply_transformation(img, matrix):
    """
    Apply a given transformation matrix to an image.
    """
    h, w = img.shape
    transformed_img = cv2.warpAffine(img, matrix, (w, h))
    return transformed_img


def compute_repeatability(keypoints1, keypoints2, threshold=5):
    """
    Compute the repeatability between two sets of keypoints.
    """
    points1 = np.array([kp.pt for kp in keypoints1])
    points2 = np.array([kp.pt for kp in keypoints2])

    # Compute distances between all points
    distances = np.linalg.norm(points1[:, None, :] - points2[None, :, :], axis=2)

    # Count matches within the threshold
    matches = (distances < threshold).sum(axis=1) > 0
    repeatability = matches.sum() / len(keypoints1) if len(keypoints1) > 0 else 0
    return repeatability

def compute_localization_error(keypoints1, keypoints2):
    """
    Compute the localization error between two sets of keypoints.
    """
    points1 = np.array([kp.pt for kp in keypoints1])
    points2 = np.array([kp.pt for kp in keypoints2])

    # Compute distances between all points
    distances = np.linalg.norm(points1[:, None, :] - points2[None, :, :], axis=2)

    # Find closest match for each keypoint
    min_distances = distances.min(axis=1)
    avg_error = min_distances.mean() if len(min_distances) > 0 else float('inf')
    return avg_error


def get_rotation_matrix(angle_degrees):
    """
    Generate an affine transformation matrix for rotation by a specified angle.

    Parameters:
    angle_degrees (float): Angle of rotation in degrees.

    Returns:
    np.ndarray: 2x3 rotation matrix.
    """
    # Convert angle from degrees to radians
    theta = np.radians(angle_degrees)

    # Compute cosine and sine of the angle
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Construct the affine transformation matrix for rotation
    rotation_matrix = np.float32([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0]
    ])

    return rotation_matrix


def scale_keypoints(keypoints, scale_factor):
    """
    Scale keypoints by a given scale factor.
    """
    scaled_keypoints = [
        cv2.KeyPoint(kp.pt[0] * scale_factor, kp.pt[1] * scale_factor, kp.size) for kp in keypoints
    ]
    return scaled_keypoints


def scale_image(img, scale_factor):
    """
    Scale an image by a given scale factor.
    """
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    scaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return scaled_img


def apply_gaussian_filter(img, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian filter to an image.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_gaussian_to_filtered_keypoints(img, keypoints, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian filter around each keypoint.

    Args:
    - img: Original image
    - keypoints: List of keypoints
    - kernel_size: Size of Gaussian kernel
    - sigma: Standard deviation of Gaussian kernel

    Returns:
    - Filtered keypoints
    """
    filtered_keypoints = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # Extract local patch around keypoint
        patch_size = max(kernel_size[0], kernel_size[1])
        half_patch = patch_size // 2

        # Ensure patch is within image bounds
        x_start = max(0, x - half_patch)
        x_end = min(img.shape[1], x + half_patch + 1)
        y_start = max(0, y - half_patch)
        y_end = min(img.shape[0], y + half_patch + 1)

        # Extract local patch
        local_patch = img[y_start:y_end, x_start:x_end]

        # Apply Gaussian filter to local patch
        filtered_patch = cv2.GaussianBlur(local_patch, kernel_size, sigma)

        # Check if filtered patch is not empty
        if filtered_patch.size > 0:
            # Create new keypoint with filtered patch coordinates
            new_x = x_start + (x - x_start)
            new_y = y_start + (y - y_start)
            filtered_keypoints.append(cv2.KeyPoint(float(new_x), float(new_y), 1))

    return filtered_keypoints


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Add Gaussian noise to an image.

    Args:
    - image: Input image
    - mean: Mean of the Gaussian noise distribution
    - sigma: Standard deviation of the Gaussian noise distribution

    Returns:
    - Noisy image
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def add_gaussian_noise_to_keypoints(keypoints, sigma=25.0):
    """
    Add Gaussian noise to keypoint coordinates.

    Args:
    - keypoints: List of OpenCV KeyPoints
    - sigma: Standard deviation of Gaussian noise

    Returns:
    - List of noisy keypoints
    """
    noisy_keypoints = []
    for kp in keypoints:
        # Add Gaussian noise to x and y coordinates
        noise_x = np.random.normal(0, sigma)
        noise_y = np.random.normal(0, sigma)

        noisy_x = kp.pt[0] + noise_x
        noisy_y = kp.pt[1] + noise_y

        noisy_kp = cv2.KeyPoint(noisy_x, noisy_y, 1)
        noisy_keypoints.append(noisy_kp)

    return noisy_keypoints
