import cv2
from transformations import compute_repeatability, compute_localization_error, apply_rotation, get_rotation_matrix, \
    apply_transformation, calculate_function_and_duration, scale_keypoints, scale_image, \
    apply_gaussian_to_filtered_keypoints, apply_gaussian_filter, add_gaussian_noise_to_keypoints, add_gaussian_noise


def get_orb_rotation_comparison(img, angle):
    """
    Compares ORB keypoints in the original image with those in the rotated image.

    Parameters:
    - img: The input image.
    - angle: The rotation angle in degrees.

    Returns:
    - repeatability: The repeatability score between the original and rotated keypoints.
    - localization_error: The localization error between the original and rotated keypoints.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors in the original image
    keypoints_original, descriptors1 = orb.detectAndCompute(img, None)

    # Apply a rotation to the original keypoints
    rotated_keypoints = apply_rotation(keypoints_original, img.shape, angle=angle)

    # Apply a geometric transformation (affine) for rotation
    affine_matrix = get_rotation_matrix(angle)
    transformed_img = apply_transformation(img, affine_matrix)

    # Detect keypoints in the transformed (rotated) image
    keypoints_transformed, descriptors2 = orb.detectAndCompute(transformed_img, None)

    # Compare the keypoints using repeatability and localization error
    repeatability = compute_repeatability(rotated_keypoints, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(rotated_keypoints, keypoints_transformed)

    return repeatability, localization_error


def get_orb_scale_comparison(img, scale_factor):
    """
    Compares ORB keypoints in the original image with those in the scaled image.

    Parameters:
    - img: The input image.
    - scale_factor: The scale factor for resizing the image.

    Returns:
    - repeatability: The repeatability score between the original and scaled keypoints.
    - localization_error: The localization error between the original and scaled keypoints.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors in the original image
    keypoints_original, descriptors1 = orb.detectAndCompute(img, None)

    # Scenario 1: Scale the keypoints directly
    scaled_keypoints = scale_keypoints(keypoints_original, scale_factor)

    # Scale the image
    transformed_img = scale_image(img, scale_factor)

    # Detect keypoints in the scaled image
    keypoints_transformed, descriptors2 = orb.detectAndCompute(transformed_img, None)

    # Compare the keypoints using repeatability and localization error
    repeatability = compute_repeatability(scaled_keypoints, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(scaled_keypoints, keypoints_transformed)

    return repeatability, localization_error


def get_orb_gaussian_filter_comparison(img):
    """
    Compares ORB keypoints in the original image with those in a Gaussian filtered image.

    Parameters:
    - img: The input image.

    Returns:
    - repeatability: The repeatability score between the original and Gaussian filtered keypoints.
    - localization_error: The localization error between the original and Gaussian filtered keypoints.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors in the original image
    keypoints_original, descriptors1 = orb.detectAndCompute(img, None)

    # Scenario 1: Apply Gaussian filter AFTER detecting keypoints
    img_gaussian_after_alg = apply_gaussian_to_filtered_keypoints(img, keypoints_original)

    # Scenario 2: Apply Gaussian filter BEFORE detecting keypoints
    img_gaussian_before_alg = apply_gaussian_filter(img)
    keypoints_transformed, descriptors2 = orb.detectAndCompute(img_gaussian_before_alg, None)

    # Compare the keypoints using repeatability and localization error
    repeatability = compute_repeatability(img_gaussian_after_alg, keypoints_transformed, threshold=5)
    localization_error = compute_localization_error(img_gaussian_after_alg, keypoints_transformed)

    return repeatability, localization_error


def get_orb_gaussian_noise_comparison(image, noise_sigma):
    """
    Compares ORB keypoints in the original image with those in a noisy image.

    Parameters:
    - image: The input image.
    - noise_sigma: The standard deviation of the Gaussian noise to add.

    Returns:
    - repeatability: The repeatability score between the original and noisy keypoints.
    - localization_error: The localization error between the original and noisy keypoints.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors in the original image
    keypoints_original, descriptors1 = orb.detectAndCompute(image, None)

    # Scenario 1: Add Gaussian noise to the keypoints themselves
    noisy_points_after_alg = add_gaussian_noise_to_keypoints(keypoints_original, sigma=noise_sigma)

    # Add Gaussian noise to the image
    noisy_image = add_gaussian_noise(image, sigma=noise_sigma)
    noisy_points_before_alg, descriptors2 = orb.detectAndCompute(noisy_image, None)

    # Compare the keypoints using repeatability and localization error
    repeatability = compute_repeatability(noisy_points_after_alg, noisy_points_before_alg, threshold=5)
    localization_error = compute_localization_error(noisy_points_after_alg, noisy_points_before_alg)

    return repeatability, localization_error


def get_orb_results(image_obj):
    """
    Collects and computes results for ORB keypoints under different transformations.

    Parameters:
    - image_obj: The input image object.

    Returns:
    - results: A list of repeatability and localization error scores for different transformations.
    """
    return [
        calculate_function_and_duration(get_orb_rotation_comparison, image_obj, 30),
        calculate_function_and_duration(get_orb_rotation_comparison, image_obj, 70),
        calculate_function_and_duration(get_orb_scale_comparison, image_obj, 2),
        calculate_function_and_duration(get_orb_scale_comparison, image_obj, 0.5),
        calculate_function_and_duration(get_orb_gaussian_filter_comparison, image_obj),
        calculate_function_and_duration(get_orb_gaussian_noise_comparison, image_obj, 25),
        calculate_function_and_duration(get_orb_gaussian_noise_comparison, image_obj, 2),
    ]
