import cv2
from transformations import get_rotation_matrix, apply_transformation, \
    scale_image, calculate_descriptor_function_and_duration, \
    apply_gaussian_filter, add_gaussian_noise


def match_descriptors(desc1, desc2):
    """
    Matches descriptors between two images using FLANN (Fast Library for Approximate Nearest Neighbors).

    Parameters:
    - desc1: SIFT descriptors from the first image.
    - desc2: SIFT descriptors from the second image.

    Returns:
    - matches: List of FLANN-based matches.
    """
    # FLANN parameters (for fast matching)
    index_params = dict(algorithm=1, trees=10)  # Use KDTrees for matching
    search_params = dict(checks=50)  # Number of checks for the nearest neighbor search

    # Create FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform the matching between descriptors (using knnMatch)
    matches = flann.knnMatch(desc1, desc2, k=2)

    return matches


def calculate_matching_precision(matches, ratio_test_threshold=0.75):
    """
    Calculates the precision of descriptor matches using the ratio test as described in the SIFT paper.

    Parameters:
    - matches: List of FLANN-based matches.
    - ratio_test_threshold: Ratio threshold to filter good matches (default is 0.75).

    Returns:
    - precision: The matching precision (True Positives / Total Matches).
    """
    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test_threshold * n.distance:
            good_matches.append(m)

    # Calculate precision: Precision = TP / (TP + FP)
    TP = len(good_matches)
    FP = len(matches) - TP  # False positives = Total matches - True positives
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    return precision


def sift_descriptor(image):
    """
    Extracts SIFT keypoints and descriptors from an image.

    Parameters:
    - image: Input image (NumPy array).

    Returns:
    - keypoints: List of keypoints detected in the image.
    - descriptors: Descriptors for the keypoints detected.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def sift_descriptor_with_rotation(img, angle):
    """
    Compares SIFT descriptors from an image before and after applying rotation.

    Parameters:
    - img: Input image.
    - angle: Angle of rotation in degrees.

    Returns:
    - precision: The precision of matches after rotation.
    """
    original_keypoints, original_descriptors = sift_descriptor(img)

    affine_matrix = get_rotation_matrix(angle)
    transformed_img = apply_transformation(img, affine_matrix)

    transformed_keypoints, transformed_descriptors = sift_descriptor(transformed_img)

    matches = match_descriptors(original_descriptors, transformed_descriptors)
    precision = calculate_matching_precision(matches)

    return precision


def sift_descriptor_with_scale(img, scale_factor):
    """
    Compares SIFT descriptors from an image before and after scaling.

    Parameters:
    - img: Input image.
    - scale_factor: Scaling factor for the image.

    Returns:
    - precision: The precision of matches after scaling.
    """
    original_keypoints, original_descriptors = sift_descriptor(img)

    transformed_img = scale_image(img, scale_factor)

    transformed_keypoints, transformed_descriptors = sift_descriptor(transformed_img)

    matches = match_descriptors(original_descriptors, transformed_descriptors)
    precision = calculate_matching_precision(matches)

    return precision


def sift_descriptor_with_gaussian_filter(img):
    """
    Compares SIFT descriptors from an image before and after applying Gaussian filter.

    Parameters:
    - img: Input image.

    Returns:
    - precision: The precision of matches after Gaussian filtering.
    """
    original_keypoints, original_descriptors = sift_descriptor(img)

    transformed_img = apply_gaussian_filter(img)

    transformed_keypoints, transformed_descriptors = sift_descriptor(transformed_img)

    matches = match_descriptors(original_descriptors, transformed_descriptors)
    precision = calculate_matching_precision(matches)

    return precision


def sift_descriptor_with_gaussian_noise(img, noise_sigma):
    """
    Compares SIFT descriptors from an image before and after adding Gaussian noise.

    Parameters:
    - img: Input image.
    - noise_sigma: Standard deviation of Gaussian noise to add.

    Returns:
    - precision: The precision of matches after adding Gaussian noise.
    """
    original_keypoints, original_descriptors = sift_descriptor(img)

    transformed_img = add_gaussian_noise(img, sigma=noise_sigma)

    transformed_keypoints, transformed_descriptors = sift_descriptor(transformed_img)

    matches = match_descriptors(original_descriptors, transformed_descriptors)
    precision = calculate_matching_precision(matches)

    return precision


def get_sift_descriptor_results(image_obj):
    """
    Collects and computes the SIFT descriptor results under various transformations.

    Parameters:
    - image_obj: The input image object.

    Returns:
    - results: A list of precision scores and computation times for different transformations.
    """
    return [
        calculate_descriptor_function_and_duration(sift_descriptor_with_rotation, image_obj, 30),
        calculate_descriptor_function_and_duration(sift_descriptor_with_rotation, image_obj, 70),
        calculate_descriptor_function_and_duration(sift_descriptor_with_scale, image_obj, 2),
        calculate_descriptor_function_and_duration(sift_descriptor_with_scale, image_obj, 0.5),
        calculate_descriptor_function_and_duration(sift_descriptor_with_gaussian_filter, image_obj),
        calculate_descriptor_function_and_duration(sift_descriptor_with_gaussian_noise, image_obj, 25),
        calculate_descriptor_function_and_duration(sift_descriptor_with_gaussian_noise, image_obj, 2),
    ]
