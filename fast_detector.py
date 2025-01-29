import cv2
import numpy as np
from matplotlib import pyplot as plt

# Importing necessary utility functions for transformations and metrics
from transformations import calculate_function_and_duration, compute_localization_error, compute_repeatability, \
    add_gaussian_noise_to_keypoints, apply_gaussian_filter, apply_gaussian_to_filtered_keypoints, scale_image, \
    scale_keypoints, apply_transformation, get_rotation_matrix, apply_rotation, add_gaussian_noise


def get_fast_rotation_comparison(img, angle):
    """
    Compares FAST detector performance after applying rotation to the image.

    Parameters:
    - img: The input image on which the FAST detector will be applied.
    - angle: The rotation angle (in degrees) to apply to the keypoints.

    Returns:
    - repeatability: The repeatability score between rotated and transformed keypoints.
    - localization_error: The localization error between rotated and transformed keypoints.
    """
    # Create a FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Detect keypoints in the original image
    keypoints_original = fast.detect(img, None)

    # Apply a rotation to the keypoints
    rotated_keypoints = apply_rotation(keypoints_original, img.shape, angle=angle)

    # Apply a geometric transformation (e.g., affine) to the image
    affine_matrix = get_rotation_matrix(angle)  # Example affine transformation
    transformed_img = apply_transformation(img, affine_matrix)

    # Detect keypoints in the rotated image
    keypoints_transformed = fast.detect(transformed_img, None)

    # Compute the repeatability and localization error
    repeatability = compute_repeatability(rotated_keypoints, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(rotated_keypoints, keypoints_transformed)

    return repeatability, localization_error


def get_fast_scale_comparison(img, scale_factor):
    """
    Compares FAST detector performance after applying scaling to the image.

    Parameters:
    - img: The input image on which the FAST detector will be applied.
    - scale_factor: The scaling factor to apply to the keypoints and image.

    Returns:
    - repeatability: The repeatability score between scaled and transformed keypoints.
    - localization_error: The localization error between scaled and transformed keypoints.
    """
    # Create a FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Detect keypoints in the original image
    keypoints_original = fast.detect(img, None)

    # Scale the keypoints according to the scale factor
    scaled_keypoints = scale_keypoints(keypoints_original, scale_factor)

    # Apply scaling to the image
    transformed_img = scale_image(img, scale_factor)

    # Detect keypoints in the scaled image
    keypoints_transformed = fast.detect(transformed_img, None)

    # Compute the repeatability and localization error
    repeatability = compute_repeatability(scaled_keypoints, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(scaled_keypoints, keypoints_transformed)

    return repeatability, localization_error


def get_fast_gaussian_filter_comparison(img):
    """
    Compares FAST detector performance after applying a Gaussian filter to the image.

    Parameters:
    - img: The input image on which the FAST detector will be applied.

    Returns:
    - repeatability: The repeatability score between filtered and transformed keypoints.
    - localization_error: The localization error between filtered and transformed keypoints.
    """
    # Create a FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Detect keypoints in the original image
    keypoints_original = fast.detect(img, None)

    # Apply Gaussian filter and transform keypoints
    img_gaussian_after_alg = apply_gaussian_to_filtered_keypoints(img, keypoints_original)

    # Apply a Gaussian filter to the image
    img_gaussian_before_alg = apply_gaussian_filter(img)

    # Detect keypoints in the filtered image
    keypoints_transformed = fast.detect(img_gaussian_before_alg, None)

    # Compute the repeatability and localization error
    repeatability = compute_repeatability(img_gaussian_after_alg, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(img_gaussian_after_alg, keypoints_transformed)

    return repeatability, localization_error


def get_fast_gaussian_noise_comparison(image, noise_sigma):
    """
    Compares FAST detector performance after applying Gaussian noise to the image.

    Parameters:
    - image: The input image on which the FAST detector will be applied.
    - noise_sigma: The standard deviation of the Gaussian noise to be added.

    Returns:
    - repeatability: The repeatability score between noisy and transformed keypoints.
    - localization_error: The localization error between noisy and transformed keypoints.
    """
    # Create a FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Detect keypoints in the original image
    keypoints_original = fast.detect(image, None)

    # Add Gaussian noise to the keypoints
    noisy_points_after_alg = add_gaussian_noise_to_keypoints(keypoints_original, sigma=noise_sigma)

    # Add Gaussian noise to the image
    noisy_image = add_gaussian_noise(image, sigma=noise_sigma)

    # Detect keypoints in the noisy image
    noisy_points_before_alg = fast.detect(noisy_image, None)

    # Compute the repeatability and localization error
    repeatability = compute_repeatability(noisy_points_after_alg, noisy_points_before_alg, threshold=5)
    localization_error = compute_localization_error(noisy_points_after_alg, noisy_points_before_alg)

    return repeatability, localization_error


def get_fast_results(image_obj):
    """
    Collects and computes results from FAST detector for various transformations.

    Parameters:
    - image_obj: The input image on which the FAST detector will be applied.

    Returns:
    - A list of results for different transformations (rotation, scaling, Gaussian filter, noise).
    """
    # Calculate function and duration for each transformation (rotation, scale, Gaussian filter, Gaussian noise)
    return [
        calculate_function_and_duration(get_fast_rotation_comparison, image_obj, 30),
        calculate_function_and_duration(get_fast_rotation_comparison, image_obj, 70),
        calculate_function_and_duration(get_fast_scale_comparison, image_obj, 2),
        calculate_function_and_duration(get_fast_scale_comparison, image_obj, 0.5),
        calculate_function_and_duration(get_fast_gaussian_filter_comparison, image_obj),
        calculate_function_and_duration(get_fast_gaussian_noise_comparison, image_obj, 25),
        calculate_function_and_duration(get_fast_gaussian_noise_comparison, image_obj, 2),
    ]
