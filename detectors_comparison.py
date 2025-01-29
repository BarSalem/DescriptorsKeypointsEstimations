from copy import deepcopy
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Importing functions to get results from different feature detectors
from fast_detector import get_fast_results
from harris_corner_detector import get_harris_corner_results
from orb_detector import get_orb_results
from sift_detector import get_sift_results

# Order of transformations for displaying in plots
ALGORITHMS_ORDER = ["Rotation\n30", "Rotation\n70", "Scale\n2", "Scale\n0.5", "Gaussian\nFilter",
                    "Gaussian\nNoise\nSigma=25", "Gaussian\nNoise\nSigma=2"]

# Initial structure to store results (repeatability, duration, localization error) for each transformation
TRANSFORMATIONS_VALUES = [{'repeatability': 0, "duration": 0, "localization_error": 0},
                          {'repeatability': 0, "duration": 0, "localization_error": 0},
                          {'repeatability': 0, "duration": 0, "localization_error": 0},
                          {'repeatability': 0, "duration": 0, "localization_error": 0},
                          {'repeatability': 0, "duration": 0, "localization_error": 0},
                          {'repeatability': 0, "duration": 0, "localization_error": 0},
                          {'repeatability': 0, "duration": 0, "localization_error": 0}]

# Dictionary to store cumulative results for each algorithm (ORB, SIFT, FAST, Harris)
ALGS_VALUES = {
    "orb": deepcopy(TRANSFORMATIONS_VALUES),
    "sift": deepcopy(TRANSFORMATIONS_VALUES),
    "fast": deepcopy(TRANSFORMATIONS_VALUES),
    "harris": deepcopy(TRANSFORMATIONS_VALUES),
}


def update_total_values_dict(orb, sift, fast, harris):
    """
    Updates the cumulative results for all four algorithms (ORB, SIFT, FAST, Harris)
    by adding the results of the current image to the existing total values.

    Parameters:
    - orb: Results from the ORB detector.
    - sift: Results from the SIFT detector.
    - fast: Results from the FAST detector.
    - harris: Results from the Harris corner detector.
    """
    for index, dict_val in enumerate(orb):
        ALGS_VALUES['orb'][index]['repeatability'] += dict_val['repeatability'] * 0.1
        ALGS_VALUES['orb'][index]['localization_error'] += dict_val['localization_error'] * 0.1
        ALGS_VALUES['orb'][index]['duration'] += dict_val['duration'] * 0.1
    for index, dict_val in enumerate(sift):
        ALGS_VALUES['sift'][index]['repeatability'] += dict_val['repeatability'] * 0.1
        ALGS_VALUES['sift'][index]['localization_error'] += dict_val['localization_error'] * 0.1
        ALGS_VALUES['sift'][index]['duration'] += dict_val['duration'] * 0.1
    for index, dict_val in enumerate(fast):
        ALGS_VALUES['fast'][index]['repeatability'] += dict_val['repeatability'] * 0.1
        ALGS_VALUES['fast'][index]['localization_error'] += dict_val['localization_error'] * 0.1
        ALGS_VALUES['fast'][index]['duration'] += dict_val['duration'] * 0.1
    for index, dict_val in enumerate(harris):
        ALGS_VALUES['harris'][index]['repeatability'] += dict_val['repeatability'] * 0.1
        ALGS_VALUES['harris'][index]['localization_error'] += dict_val['localization_error'] * 0.1
        ALGS_VALUES['harris'][index]['duration'] += dict_val['duration'] * 0.1


def compare_and_show_results(image_path, show_image=True):
    """
    Compares the results of four feature detectors (ORB, SIFT, FAST, Harris) for a given image.
    Displays bar plots for duration, repeatability, and localization error, and saves the plots.

    Parameters:
    - image_path: Path to the input image to be processed.
    - show_image: Boolean to determine whether to display the image in the plot.
    """
    # Read the image in grayscale
    image_obj = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get feature detection results for each detector
    sift = get_sift_results(image_obj)
    fast = get_fast_results(image_obj)
    harris = get_harris_corner_results(image_obj)
    orb = get_orb_results(image_obj)

    # Update cumulative results with current image's data
    update_total_values_dict(orb, sift, fast, harris)

    # Prepare data for plotting
    lists = [sift, orb, harris, fast]
    algorithms = ['SIFT', 'ORB', 'HARRIS', 'FAST']

    # Indices for comparing results
    indices = range(len(sift))

    # Keys for the metrics to compare: 'duration', 'localization_error', 'repeatability'
    keys = ['duration', 'localization_error', 'repeatability']

    # Create the figure with subplots for each metric and image
    fig = plt.figure(figsize=(24, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.5, 1.5, 1.5])

    # Subplot for displaying the base image
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image_obj, cmap='gray')
    ax_img.set_title('Base Image', fontsize=14)
    ax_img.axis('off')

    # Create subplots for each metric (duration, repeatability, localization error)
    axes = [fig.add_subplot(gs[i + 1]) for i in range(len(keys))]

    bar_width = 0.15  # Width of bars in the bar charts

    for i, key in enumerate(keys):
        # Extract the metric values for each detector and each index
        data = [[lst[idx][key] for lst in lists] for idx in indices]

        # Plot the data as a grouped bar chart
        x = np.arange(len(indices))  # X positions for bars
        ax = axes[i]  # Current subplot
        for j, algorithm in enumerate(algorithms):
            values = [data[idx][j] for idx in range(len(indices))]
            ax.bar(x + j * bar_width, values, width=bar_width, label=algorithm)

        # Customize the subplot with titles, labels, and legends
        ax.set_title(f'{key.capitalize()} Comparison', fontsize=14)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_xticks(x + bar_width * (len(algorithms) / 2 - 0.5))  # Center tick labels
        ax.set_xticklabels([ALGORITHMS_ORDER[i] for i in indices])
        if i == 0:  # Add y-label only for the first subplot (duration)
            ax.set_ylabel(key.capitalize(), fontsize=12)
        ax.legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot with comparison results
    image_name = image_path.replace("./images/", "").replace(".png", "").replace(".jpg", "")
    output_path = f"./comparison_results_detectors/comparison_plot_{image_name}.png"
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')

    # Optionally show the plot
    if show_image:
        plt.show()


def save_summary_run_plot(show_plot=False):
    """
    Saves a summary plot comparing the cumulative results of all feature detectors (ORB, SIFT, FAST, Harris)
    across all transformations. The plot includes duration, repeatability, and localization error metrics.

    Parameters:
    - show_plot: Boolean to control whether to display the summary plot.
    """
    data = deepcopy(ALGS_VALUES)

    # Indices for the transformations
    indices = range(7)
    algorithms = ['Fast', 'Harris', 'ORB', 'SIFT']

    # Collect duration, repeatability, and localization error values for each algorithm
    duration_values = []
    repeatability_values = []
    localization_error_values = []

    for idx in indices:
        # Collect values for each metric and algorithm
        duration_values.append([data['fast'][idx]['duration'], data['harris'][idx]['duration'],
                                data['orb'][idx]['duration'], data['sift'][idx]['duration']])
        repeatability_values.append([data['fast'][idx]['repeatability'], data['harris'][idx]['repeatability'],
                                     data['orb'][idx]['repeatability'], data['sift'][idx]['repeatability']])
        localization_error_values.append([data['fast'][idx]['localization_error'],
                                          data['harris'][idx]['localization_error'],
                                          data['orb'][idx]['localization_error'],
                                          data['sift'][idx]['localization_error']])

    # Settings for the bar chart
    bar_width = 0.2
    x = np.arange(len(indices))

    # Create the plot with three subplots: duration, repeatability, and localization error
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot bars for each metric
    for i, algorithm in enumerate(algorithms):
        ax1.bar(x + i * bar_width, [duration_values[idx][i] for idx in indices], bar_width, label=algorithm)
        ax2.bar(x + i * bar_width, [repeatability_values[idx][i] for idx in indices], bar_width, label=algorithm)
        ax3.bar(x + i * bar_width, [localization_error_values[idx][i] for idx in indices], bar_width, label=algorithm)

    # Customize the plots with titles, labels, and legends
    ax1.set_xlabel('Index', fontsize=14)
    ax1.set_ylabel('Duration (seconds)', fontsize=14)
    ax1.set_title('Duration Comparison for Each Index', fontsize=16)
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels([f"{ALGORITHMS_ORDER[i]}" for i in indices], fontsize=12)
    ax1.legend()

    ax2.set_xlabel('Index', fontsize=14)
    ax2.set_ylabel('Repeatability', fontsize=14)
    ax2.set_title('Repeatability Comparison for Each Index', fontsize=16)
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels([f"{ALGORITHMS_ORDER[i]}" for i in indices], fontsize=12)
    ax2.legend()

    ax3.set_xlabel('Index', fontsize=14)
    ax3.set_ylabel('Localization Error', fontsize=14)
    ax3.set_title('Localization Error Comparison for Each Index', fontsize=16)
    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels([f"{ALGORITHMS_ORDER[i]}" for i in indices], fontsize=12)
    ax3.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the summary plot
    output_path = './comparison_results_detectors/comparison_summary.png'
    plt.savefig(output_path, format='png', dpi=300)

    # Optionally show the plot
    if show_plot:
        plt.show()


def compare_images():
    """
    Main function to iterate over all images in the 'images' directory, run the feature detectors,
    and save comparison plots. After processing all images, a summary plot is saved.
    """
    print("Starting to run detectors experiment")
    dir_path = "./images"
    images_names = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    for image in images_names:
        print(f"Starting process image: {image}")
        compare_and_show_results(f'./images/{image}', show_image=False)

    # Save the summary comparison plot after processing all images
    save_summary_run_plot()
