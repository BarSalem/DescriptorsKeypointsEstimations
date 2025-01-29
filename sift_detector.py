import cv2
import numpy as np
from transformations import compute_repeatability, compute_localization_error, apply_rotation, get_rotation_matrix, \
    apply_transformation, calculate_function_and_duration, scale_keypoints, scale_image, \
    apply_gaussian_to_filtered_keypoints, apply_gaussian_filter, add_gaussian_noise_to_keypoints, add_gaussian_noise


def get_sift_rotation_comparison(img, angle):
    """
    Compares SIFT keypoints from an image before and after rotation.

    Parameters:
    - img: Input image (NumPy array).
    - angle: Rotation angle in degrees.

    Returns:
    - repeatability: The repeatability score of keypoints.
    - localization_error: The localization error score of keypoints.
    """
    sift = cv2.SIFT_create()

    keypoints_original, descriptors1 = sift.detectAndCompute(img, None)

    if not keypoints_original:
        return 0, 0  # Return 0 if no keypoints were detected.

    # Apply a rotation to the corners
    rotated_keypoints = apply_rotation(keypoints_original, img.shape, angle=angle)

    # Apply a geometric transformation (rotation)
    affine_matrix = get_rotation_matrix(angle)  # Example affine transformation
    transformed_img = apply_transformation(img, affine_matrix)

    # Detect corners in the transformed image
    keypoints_transformed, descriptors2 = sift.detectAndCompute(transformed_img, None)

    if not keypoints_transformed:
        return 0, 0  # Return 0 if no keypoints were detected after transformation.

    # Compare corners
    repeatability = compute_repeatability(rotated_keypoints, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(rotated_keypoints, keypoints_transformed)

    return repeatability, localization_error


def get_sift_scale_comparison(img, scale_factor):
    """
    Compares SIFT keypoints from an image before and after scaling.

    Parameters:
    - img: Input image (NumPy array).
    - scale_factor: Scaling factor for the image.

    Returns:
    - repeatability: The repeatability score of keypoints.
    - localization_error: The localization error score of keypoints.
    """
    sift = cv2.SIFT_create()

    keypoints_original, descriptors1 = sift.detectAndCompute(img, None)

    if not keypoints_original:
        return 0, 0  # Return 0 if no keypoints were detected.

    # Scale the keypoints
    scaled_keypoints = scale_keypoints(keypoints_original, scale_factor)

    # Scale the image
    transformed_img = scale_image(img, scale_factor)

    # Detect corners in the transformed image
    keypoints_transformed, descriptors2 = sift.detectAndCompute(transformed_img, None)

    if not keypoints_transformed:
        return 0, 0  # Return 0 if no keypoints were detected after transformation.

    # Compare corners
    repeatability = compute_repeatability(scaled_keypoints, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(scaled_keypoints, keypoints_transformed)

    return repeatability, localization_error


def get_sift_gaussian_filter_comparison(img):
    """
    Compares SIFT keypoints from an image before and after applying Gaussian filtering.

    Parameters:
    - img: Input image (NumPy array).

    Returns:
    - repeatability: The repeatability score of keypoints.
    - localization_error: The localization error score of keypoints.
    """
    sift = cv2.SIFT_create()

    keypoints_original, descriptors1 = sift.detectAndCompute(img, None)

    if not keypoints_original:
        return 0, 0  # Return 0 if no keypoints were detected.

    # Apply Gaussian filter to the image and keypoints
    img_gaussian_after_alg = apply_gaussian_to_filtered_keypoints(img, keypoints_original)

    img_gaussian_before_alg = apply_gaussian_filter(img)

    # Detect corners in the transformed image
    keypoints_transformed, descriptors2 = sift.detectAndCompute(img_gaussian_before_alg, None)

    if not keypoints_transformed:
        return 0, 0  # Return 0 if no keypoints were detected after transformation.

    # Compare corners
    repeatability = compute_repeatability(img_gaussian_after_alg, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(img_gaussian_after_alg, keypoints_transformed)

    return repeatability, localization_error


def get_sift_gaussian_noise_comparison(image, noise_sigma):
    """
    Compares SIFT keypoints from an image before and after adding Gaussian noise.

    Parameters:
    - image: Input image (NumPy array).
    - noise_sigma: Standard deviation of Gaussian noise.

    Returns:
    - repeatability: The repeatability score of keypoints.
    - localization_error: The localization error score of keypoints.
    """
    sift = cv2.SIFT_create()

    keypoints_original, descriptors1 = sift.detectAndCompute(image, None)

    if not keypoints_original:
        return 0, 0  # Return 0 if no keypoints were detected.

    # Add noise to the keypoints and image
    noisy_points_after_alg = add_gaussian_noise_to_keypoints(keypoints_original, sigma=noise_sigma)

    noisy_image = add_gaussian_noise(image, sigma=noise_sigma)
    noisy_points_before_alg, descriptors1 = sift.detectAndCompute(noisy_image, None)

    if not noisy_points_before_alg:
        return 0, 0  # Return 0 if no keypoints were detected in noisy image.

    # Compare corners
    repeatability = compute_repeatability(noisy_points_after_alg, noisy_points_before_alg, threshold=5)
    localization_error = compute_localization_error(noisy_points_after_alg, noisy_points_before_alg)

    return repeatability, localization_error


def get_sift_results(image_obj):
    """
    Collects and computes SIFT descriptor results for various transformations.

    Parameters:
    - image_obj: Input image object (NumPy array).

    Returns:
    - results: A list of repeatability and localization error scores for different transformations.
    """
    return [
        calculate_function_and_duration(get_sift_rotation_comparison, image_obj, 30),
        calculate_function_and_duration(get_sift_rotation_comparison, image_obj, 70),
        calculate_function_and_duration(get_sift_scale_comparison, image_obj, 2),
        calculate_function_and_duration(get_sift_scale_comparison, image_obj, 0.5),
        calculate_function_and_duration(get_sift_gaussian_filter_comparison, image_obj),
        calculate_function_and_duration(get_sift_gaussian_noise_comparison, image_obj, 25),
        calculate_function_and_duration(get_sift_gaussian_noise_comparison, image_obj, 2),
    ]
