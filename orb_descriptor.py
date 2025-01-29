import cv2
from transformations import get_rotation_matrix, apply_transformation, \
    scale_image, calculate_descriptor_function_and_duration, \
    apply_gaussian_filter, add_gaussian_noise


def match_descriptors(descriptors1, descriptors2):
    """
    Matches descriptors between two images using Brute Force Matcher with Hamming distance.

    Parameters:
    - descriptors1: Descriptors from the first image.
    - descriptors2: Descriptors from the second image.

    Returns:
    - matches: Sorted list of good matches between descriptors.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors using the Brute Force Matcher
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance
    return matches


def calculate_matching_precision(matches, threshold=10):
    """
    Calculates the precision of the descriptor matching.

    Precision is the fraction of good matches to the total number of matches.
    A match is considered good if its distance is below the threshold.

    Parameters:
    - matches: List of matches between two sets of descriptors.
    - threshold: The distance threshold below which matches are considered good.

    Returns:
    - precision: The precision score (ratio of good matches to total matches).
    """
    good_matches = [m for m in matches if m.distance < threshold]
    precision = len(good_matches) / len(matches) if matches else 0  # Handle case of no matches
    return precision


def orb_descriptor(image):
    """
    Detects keypoints and computes descriptors using ORB (Oriented FAST and Rotated BRIEF).

    Parameters:
    - image: The input image.

    Returns:
    - keypoints: List of detected keypoints.
    - descriptors: List of descriptors corresponding to the keypoints.
    """
    orb = cv2.ORB_create()  # Create ORB detector
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def orb_descriptor_with_rotation(img, angle):
    """
    Evaluates the precision of ORB descriptors before and after applying a rotation.

    Parameters:
    - img: The input image.
    - angle: The rotation angle in degrees.

    Returns:
    - precision: The precision score after matching descriptors between the original and rotated image.
    """
    # Detect keypoints and compute descriptors for the original image
    original_keypoints, original_descriptors = orb_descriptor(img)

    # Apply rotation to the image and detect keypoints
    affine_matrix = get_rotation_matrix(angle)  # Get the rotation matrix
    transformed_img = apply_transformation(img, affine_matrix)
    transformed_keypoints, transformed_descriptors = orb_descriptor(transformed_img)

    # Match descriptors and calculate precision
    matches = match_descriptors(original_descriptors, transformed_descriptors)
    precision = calculate_matching_precision(matches)

    return precision


def orb_descriptor_with_scale(img, scale_factor):
    """
    Evaluates the precision of ORB descriptors before and after applying scaling.

    Parameters:
    - img: The input image.
    - scale_factor: The scaling factor for the image.

    Returns:
    - precision: The precision score after matching descriptors between the original and scaled image.
    """
    # Detect keypoints and compute descriptors for the original image
    original_keypoints, original_descriptors = orb_descriptor(img)

    # Apply scaling to the image and detect keypoints
    transformed_img = scale_image(img, scale_factor)
    transformed_keypoints, transformed_descriptors = orb_descriptor(transformed_img)

    # Match descriptors and calculate precision
    matches = match_descriptors(original_descriptors, transformed_descriptors)
    precision = calculate_matching_precision(matches)

    return precision


def orb_descriptor_with_gaussian_filter(img):
    """
    Evaluates the precision of ORB descriptors before and after applying a Gaussian filter.

    Parameters:
    - img: The input image.

    Returns:
    - precision: The precision score after matching descriptors between the original and filtered image.
    """
    # Detect keypoints and compute descriptors for the original image
    original_keypoints, original_descriptors = orb_descriptor(img)

    # Apply Gaussian filter to the image and detect keypoints
    transformed_img = apply_gaussian_filter(img)
    transformed_keypoints, transformed_descriptors = orb_descriptor(transformed_img)

    # Match descriptors and calculate precision
    matches = match_descriptors(original_descriptors, transformed_descriptors)
    precision = calculate_matching_precision(matches)

    return precision


def orb_descriptor_with_gaussian_noise(img, noise_sigma):
    """
    Evaluates the precision of ORB descriptors before and after adding Gaussian noise.

    Parameters:
    - img: The input image.
    - noise_sigma: The standard deviation of the Gaussian noise to be added.

    Returns:
    - precision: The precision score after matching descriptors between the original and noisy image.
    """
    # Detect keypoints and compute descriptors for the original image
    original_keypoints, original_descriptors = orb_descriptor(img)

    # Add Gaussian noise to the image and detect keypoints
    transformed_img = add_gaussian_noise(img, sigma=noise_sigma)
    transformed_keypoints, transformed_descriptors = orb_descriptor(transformed_img)

    # Match descriptors and calculate precision
    matches = match_descriptors(original_descriptors, transformed_descriptors)
    precision = calculate_matching_precision(matches)

    return precision


def get_orb_descriptor_results(image_obj):
    """
    Collects and computes results for ORB descriptor matching under different transformations.

    Parameters:
    - image_obj: The input image object.

    Returns:
    - results: A list of precision scores for different transformations (rotation, scaling, filtering, noise).
    """
    return [
        calculate_descriptor_function_and_duration(orb_descriptor_with_rotation, image_obj, 30),
        calculate_descriptor_function_and_duration(orb_descriptor_with_rotation, image_obj, 70),
        calculate_descriptor_function_and_duration(orb_descriptor_with_scale, image_obj, 2),
        calculate_descriptor_function_and_duration(orb_descriptor_with_scale, image_obj, 0.5),
        calculate_descriptor_function_and_duration(orb_descriptor_with_gaussian_filter, image_obj),
        calculate_descriptor_function_and_duration(orb_descriptor_with_gaussian_noise, image_obj, 25),
        calculate_descriptor_function_and_duration(orb_descriptor_with_gaussian_noise, image_obj, 2),
    ]
