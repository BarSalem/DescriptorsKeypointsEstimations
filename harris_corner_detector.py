import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importing utility functions for transformations and metrics
from transformations import apply_rotation, apply_transformation, compute_repeatability, compute_localization_error, \
    get_rotation_matrix, calculate_function_and_duration, scale_keypoints, scale_image, apply_gaussian_filter, \
    apply_gaussian_to_filtered_keypoints, add_gaussian_noise, add_gaussian_noise_to_keypoints


def harris_corner_detection(img, threshold=0.01):
    """
    Detect Harris corners in an image.

    Parameters:
    - img: The input image.
    - threshold: The threshold for corner detection.

    Returns:
    - keypoints: A list of detected keypoints (corners).
    """
    gray = np.float32(img)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)  # Dilate to highlight corners

    # Threshold to retain only the strong corners
    keypoints = np.argwhere(dst > threshold * dst.max())
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints]
    return keypoints


def get_harris_corner_detection_rotation_comparison(img, angle: int):
    """
    Compares Harris corner detection performance before and after applying a rotation.

    Parameters:
    - img: The input image.
    - angle: The rotation angle in degrees.

    Returns:
    - repeatability: The repeatability score between rotated and transformed keypoints.
    - localization_error: The localization error between rotated and transformed keypoints.
    """
    # Detect corners in the original image
    keypoints_original = harris_corner_detection(img)

    # Apply rotation to the keypoints
    rotated_keypoints = apply_rotation(keypoints_original, img.shape, angle=angle)

    # Apply rotation to the image and detect corners
    affine_matrix = get_rotation_matrix(angle)  # Rotation matrix
    transformed_img = apply_transformation(img, affine_matrix)
    keypoints_transformed = harris_corner_detection(transformed_img)

    # Compare the keypoints
    repeatability = compute_repeatability(rotated_keypoints, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(rotated_keypoints, keypoints_transformed)

    return repeatability, localization_error


def get_harris_corner_detection_scale_comparison(img, scale_factor):
    """
    Compares Harris corner detection performance before and after applying scaling.

    Parameters:
    - img: The input image.
    - scale_factor: The scaling factor for the image and keypoints.

    Returns:
    - repeatability: The repeatability score between scaled and transformed keypoints.
    - localization_error: The localization error between scaled and transformed keypoints.
    """
    # Detect corners in the original image
    keypoints_original = harris_corner_detection(img)

    # Scenario 1: Scale the keypoints first, then compare
    scaled_keypoints = scale_keypoints(keypoints_original, scale_factor)

    # Scenario 2: Scale the image, then detect corners
    scaled_img = scale_image(img, scale_factor)
    keypoints_scaled_image = harris_corner_detection(scaled_img)

    # Compare the keypoints
    repeatability = compute_repeatability(scaled_keypoints, keypoints_scaled_image, threshold=5)
    localization_error = compute_localization_error(scaled_keypoints, keypoints_scaled_image)

    return repeatability, localization_error


def get_harris_corner_detection_gaussian_filter_comparison(img):
    """
    Compares Harris corner detection performance before and after applying a Gaussian filter.

    Parameters:
    - img: The input image.

    Returns:
    - repeatability: The repeatability score between filtered and transformed keypoints.
    - localization_error: The localization error between filtered and transformed keypoints.
    """
    # Scenario 1: Apply Gaussian filter to the image first, then detect corners
    keypoints_original = harris_corner_detection(img)
    img_gaussian_after = apply_gaussian_to_filtered_keypoints(img, keypoints_original)  # Apply Gaussian filter after detecting corners

    # Scenario 2: Apply Gaussian filter to the image, then detect corners
    img_gaussian_before = apply_gaussian_filter(img)
    keypoints_gaussian_before = harris_corner_detection(img_gaussian_before)

    # Compare the keypoints
    repeatability = compute_repeatability(img_gaussian_after, keypoints_gaussian_before, threshold=5)
    localization_error = compute_localization_error(img_gaussian_after, keypoints_gaussian_before)

    return repeatability, localization_error


def get_harris_corner_detection_gaussian_noise_comparison(image, noise_sigma=25):
    """
    Compares Harris corner detection performance before and after adding Gaussian noise.

    Parameters:
    - image: The input image.
    - noise_sigma: The standard deviation for Gaussian noise.

    Returns:
    - repeatability: The repeatability score between noisy and transformed keypoints.
    - localization_error: The localization error between noisy and transformed keypoints.
    """
    # Detect keypoints in the original image
    original_keypoints = harris_corner_detection(image)

    # Add Gaussian noise to the keypoints
    noisy_keypoints_1 = add_gaussian_noise_to_keypoints(original_keypoints, sigma=noise_sigma)

    # Add Gaussian noise to the image and detect corners
    noisy_image = add_gaussian_noise(image, sigma=noise_sigma)
    noisy_keypoints_2 = harris_corner_detection(noisy_image)

    # Compare the keypoints
    repeatability = compute_repeatability(noisy_keypoints_1, noisy_keypoints_2, threshold=5)
    localization_error = compute_localization_error(noisy_keypoints_1, noisy_keypoints_2)

    return repeatability, localization_error


def get_harris_corner_results(image_obj):
    """
    Collects and computes results for Harris corner detection under different transformations.

    Parameters:
    - image_obj: The input image.

    Returns:
    - A list of results for different transformations (rotation, scaling, Gaussian filter, noise).
    """
    return [
        calculate_function_and_duration(get_harris_corner_detection_rotation_comparison, image_obj, 30),
        calculate_function_and_duration(get_harris_corner_detection_rotation_comparison, image_obj, 70),
        calculate_function_and_duration(get_harris_corner_detection_scale_comparison, image_obj, 2),
        calculate_function_and_duration(get_harris_corner_detection_scale_comparison, image_obj, 0.5),
        calculate_function_and_duration(get_harris_corner_detection_gaussian_filter_comparison, image_obj),
        calculate_function_and_duration(get_harris_corner_detection_gaussian_noise_comparison, image_obj, 25),
        calculate_function_and_duration(get_harris_corner_detection_gaussian_noise_comparison, image_obj, 2),
    ]
