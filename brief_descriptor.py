import cv2

# Importing functions from transformations module to apply geometric transformations and filters
from transformations import get_rotation_matrix, apply_transformation, \
    scale_image, calculate_descriptor_function_and_duration, \
    apply_gaussian_filter, add_gaussian_noise


def match_descriptors(desc1, desc2, ratio_thresh=0.75):
    """
    Matches descriptors between two images using the BFMatcher and ratio test.

    Parameters:
    - desc1: Descriptors of the first image.
    - desc2: Descriptors of the second image.
    - ratio_thresh: Ratio threshold for filtering good matches.

    Returns:
    - good_matches: List of good matches after applying the ratio test.
    """
    # Initialize the Brute Force Matcher with Hamming distance (used for binary descriptors)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Check if descriptors are valid (not None or empty)
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    # Find 2 nearest matches for each descriptor
    matches = bf_matcher.knnMatch(desc1, desc2, k=2)

    # Apply the ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:  # Ensure we found 2 matches
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:  # Keep the match if the ratio is below the threshold
                good_matches.append(m)

    return good_matches


def calculate_matching_precision(matches, num_keypoints):
    """
    Calculates the precision of matching descriptors by dividing the number of good matches
    by the total number of keypoints.

    Parameters:
    - matches: List of good matches.
    - num_keypoints: The number of keypoints in the first image.

    Returns:
    - precision: The precision of the descriptor matching.
    """
    num_matches = len(matches)
    precision = num_matches / num_keypoints if num_keypoints > 0 else 0  # Avoid division by zero
    return precision


def brief_descriptor(image):
    """
    Computes BRIEF descriptors based on FAST keypoints detected in the image.

    Parameters:
    - image: The input image for feature detection.

    Returns:
    - keypoints: List of keypoints detected by FAST.
    - descriptors: BRIEF descriptors corresponding to the keypoints.
    """
    # Initialize the FAST feature detector with a sensitivity threshold of 20
    fast_detector = cv2.FastFeatureDetector_create(
        threshold=20,  # Increased sensitivity for better keypoint detection
        nonmaxSuppression=True  # Suppress non-maximum keypoints to avoid redundant detections
    )

    # Initialize the BRIEF descriptor extractor with a lower threshold for better descriptor detection
    brief_extractor = cv2.BRISK_create(
        thresh=10,  # Lower threshold to detect more keypoints
        octaves=0,  # Use a single scale for BRIEF-like behavior
        patternScale=1.0  # Standard pattern scale
    )

    # Detect keypoints using the FAST detector
    keypoints = fast_detector.detect(image)

    # Compute BRIEF descriptors for the detected keypoints
    keypoints, descriptors = brief_extractor.compute(image, keypoints)

    return keypoints, descriptors


def brief_descriptor_with_rotation(img, angle):
    """
    Computes BRIEF descriptors for an image with rotated keypoints and matches them with the original.

    Parameters:
    - img: The input image.
    - angle: The angle to rotate the image.

    Returns:
    - precision: The precision of descriptor matching after rotation.
    """
    # Get the original keypoints and descriptors
    original_keypoints, original_descriptors = brief_descriptor(img)

    # Get the rotation matrix and apply it to the image
    affine_matrix = get_rotation_matrix(angle)  # Example of rotation transformation
    transformed_img = apply_transformation(img, affine_matrix)

    # Compute descriptors for the transformed image
    transformed_keypoints, transformed_descriptors = brief_descriptor(transformed_img)

    # Match descriptors between the original and transformed images
    matches = match_descriptors(original_descriptors, transformed_descriptors)

    # Calculate precision of the matching process
    precision = calculate_matching_precision(matches, len(original_keypoints))
    return precision


def brief_descriptor_with_scale(img, scale_factor):
    """
    Computes BRIEF descriptors for an image with scaled keypoints and matches them with the original.

    Parameters:
    - img: The input image.
    - scale_factor: The factor by which to scale the image.

    Returns:
    - precision: The precision of descriptor matching after scaling.
    """
    # Get the original keypoints and descriptors
    original_keypoints, original_descriptors = brief_descriptor(img)

    # Apply the scaling transformation to the image
    transformed_img = scale_image(img, scale_factor)

    # Compute descriptors for the scaled image
    transformed_keypoints, transformed_descriptors = brief_descriptor(transformed_img)

    # Match descriptors between the original and scaled images
    matches = match_descriptors(original_descriptors, transformed_descriptors)

    # Calculate precision of the matching process
    precision = calculate_matching_precision(matches, len(original_keypoints))
    return precision


def brief_descriptor_with_gaussian_filter(img):
    """
    Computes BRIEF descriptors for an image with Gaussian filter applied and matches them with the original.

    Parameters:
    - img: The input image.

    Returns:
    - precision: The precision of descriptor matching after Gaussian filtering.
    """
    # Get the original keypoints and descriptors
    original_keypoints, original_descriptors = brief_descriptor(img)

    # Apply Gaussian filter to the image (smoothes the image)
    transformed_img = apply_gaussian_filter(img)

    # Compute descriptors for the filtered image
    transformed_keypoints, transformed_descriptors = brief_descriptor(transformed_img)

    # Match descriptors between the original and filtered images
    matches = match_descriptors(original_descriptors, transformed_descriptors)

    # Calculate precision of the matching process
    precision = calculate_matching_precision(matches, len(original_keypoints))
    return precision


def brief_descriptor_with_gaussian_noise(img, noise_sigma):
    """
    Computes BRIEF descriptors for an image with Gaussian noise added and matches them with the original.

    Parameters:
    - img: The input image.
    - noise_sigma: The standard deviation of the Gaussian noise to add.

    Returns:
    - precision: The precision of descriptor matching after adding Gaussian noise.
    """
    # Get the original keypoints and descriptors
    original_keypoints, original_descriptors = brief_descriptor(img)

    # Add Gaussian noise to the image
    transformed_img = add_gaussian_noise(img, sigma=noise_sigma)

    # Compute descriptors for the noisy image
    transformed_keypoints, transformed_descriptors = brief_descriptor(transformed_img)

    # Match descriptors between the original and noisy images
    matches = match_descriptors(original_descriptors, transformed_descriptors)

    # Calculate precision of the matching process
    precision = calculate_matching_precision(matches, len(original_keypoints))
    return precision


def get_brief_descriptor_results(image_obj):
    """
    Computes the results for descriptor matching under various transformations and measures the performance.

    Parameters:
    - image_obj: The image object on which transformations are applied.

    Returns:
    - A list of results from the descriptor matching functions for various transformations.
    """
    return [
        # Calculate results for rotation with different angles
        calculate_descriptor_function_and_duration(brief_descriptor_with_rotation, image_obj, 30),
        calculate_descriptor_function_and_duration(brief_descriptor_with_rotation, image_obj, 70),

        # Calculate results for scaling with different scale factors
        calculate_descriptor_function_and_duration(brief_descriptor_with_scale, image_obj, 2),
        calculate_descriptor_function_and_duration(brief_descriptor_with_scale, image_obj, 0.5),

        # Calculate results for Gaussian filtering
        calculate_descriptor_function_and_duration(brief_descriptor_with_gaussian_filter, image_obj),

        # Calculate results for adding Gaussian noise with different noise levels
        calculate_descriptor_function_and_duration(brief_descriptor_with_gaussian_noise, image_obj, 25),
        calculate_descriptor_function_and_duration(brief_descriptor_with_gaussian_noise, image_obj, 2),
    ]
