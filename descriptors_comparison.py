from copy import deepcopy
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Importing functions to get descriptor results for different algorithms
from brief_descriptor import get_brief_descriptor_results
from orb_descriptor import get_orb_descriptor_results
from sift_descriptor import get_sift_descriptor_results

# Predefined order of transformation types for plotting
ALGORITHMS_ORDER = ["Rotation\n30", "Rotation\n70", "Scale\n2", "Scale\n0.5", "Gaussian\nFilter",
                    "Gaussian\nNoise\nSigma=25",
                    "Gaussian\nNoise\nSigma=2"]

# Default structure to store precision and duration for each transformation
TRANSFORMATIONS_VALUES = [{'precision': 0, "duration": 0}, {'precision': 0, "duration": 0},
                          {'precision': 0, "duration": 0}, {'precision': 0, "duration": 0},
                          {'precision': 0, "duration": 0}, {'precision': 0, "duration": 0},
                          {'precision': 0, "duration": 0}]

# Dictionary to store results for each algorithm: SIFT, ORB, and BRIEF
ALGS_VALUES = {
    "orb": deepcopy(TRANSFORMATIONS_VALUES),
    "sift": deepcopy(TRANSFORMATIONS_VALUES),
    "brief": deepcopy(TRANSFORMATIONS_VALUES),
}


def update_total_values_dict(orb, sift, brief):
    """
    Updates the total values dictionary for each algorithm (ORB, SIFT, and BRIEF)
    by adding the current precision and duration results to the cumulative ones.

    Parameters:
    - orb: Precision and duration values for the ORB algorithm.
    - sift: Precision and duration values for the SIFT algorithm.
    - brief: Precision and duration values for the BRIEF algorithm.
    """
    for index, dict_val in enumerate(orb):
        ALGS_VALUES['orb'][index]['precision'] += dict_val['precision'] * 0.1
        ALGS_VALUES['orb'][index]['duration'] += dict_val['duration'] * 0.1
    for index, dict_val in enumerate(sift):
        ALGS_VALUES['sift'][index]['precision'] += dict_val['precision'] * 0.1
        ALGS_VALUES['sift'][index]['duration'] += dict_val['duration'] * 0.1
    for index, dict_val in enumerate(brief):
        ALGS_VALUES['brief'][index]['precision'] += dict_val['precision'] * 0.1
        ALGS_VALUES['brief'][index]['duration'] += dict_val['duration'] * 0.1


def compare_and_show_results(image_path, show_image=True):
    """
    Compares descriptor results for three algorithms (SIFT, ORB, BRIEF) on the given image
    and displays the results in histograms and a plot.

    Parameters:
    - image_path: Path to the input image.
    - show_image: Boolean to control whether to display the image in the plot.
    """
    # Load the image in grayscale
    image_obj = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get descriptor results for the image using three different algorithms
    sift = get_sift_descriptor_results(image_obj)
    orb = get_orb_descriptor_results(image_obj)
    brief = get_brief_descriptor_results(image_obj)

    # Update total values for all algorithms
    update_total_values_dict(orb, sift, brief)

    # Prepare data for plotting (store the results of each algorithm)
    lists = [sift, orb, brief]
    algorithms = ['SIFT', 'ORB', 'BRIEF']

    # Indices of transformations (used for labeling on the x-axis)
    indices = range(len(sift))

    # Keys to compare: 'precision' and 'duration'
    keys = ['duration', 'precision']

    # Create a figure for plotting with a grid layout
    fig = plt.figure(figsize=(24, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1.5])

    # First subplot for displaying the base image
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image_obj, cmap='gray')
    ax_img.set_title('Base Image', fontsize=14)
    ax_img.axis('off')  # Hide axes for better visualization

    # Create additional subplots for precision and duration histograms
    axes = [fig.add_subplot(gs[i + 1]) for i in range(len(keys))]

    bar_width = 0.15  # Bar width for the grouped bar charts

    for i, key in enumerate(keys):
        # Extract data for the current key (precision or duration)
        data = [[lst[idx][key] for lst in lists] for idx in indices]

        # Plot the grouped bar charts
        x = np.arange(len(indices))  # X-axis positions for the bars
        ax = axes[i]  # Current subplot
        for j, algorithm in enumerate(algorithms):
            values = [data[idx][j] for idx in range(len(indices))]
            ax.bar(x + j * bar_width, values, width=bar_width, label=algorithm)

        # Customize the subplot (title, labels, legend, etc.)
        ax.set_title(f'{key.capitalize()} Comparison', fontsize=14)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_xticks(x + bar_width * (len(algorithms) / 2 - 0.5))  # Center the x-tick labels
        ax.set_xticklabels([ALGORITHMS_ORDER[i] for i in indices])
        if i == 0:  # Add y-label only for the first subplot (precision)
            ax.set_ylabel(key.capitalize(), fontsize=12)
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Create an output path for saving the comparison plot
    image_name = image_path.replace("./images/", "").replace(".png", "").replace(".jpg", "")
    output_path = f"./comparison_results_descriptors/comparison_plot_{image_name}.png"
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')

    if show_image:
        # Optionally show the plot if show_image is True
        plt.show()


def save_summary_run_plot(show_plot=False):
    """
    Saves a summary plot comparing the precision and duration of the three algorithms (SIFT, ORB, BRIEF)
    across all transformations and optionally shows the plot.

    Parameters:
    - show_plot: Boolean to control whether to display the summary plot.
    """
    data = deepcopy(ALGS_VALUES)
    # Prepare data for plotting (7 indices corresponding to different transformations)
    indices = range(7)
    algorithms = ['Brief', 'ORB', 'SIFT']

    # Collect duration and precision values for each algorithm
    duration_values = []
    precision_values = []

    for idx in indices:
        # Extract the duration and precision for each algorithm at each index
        duration_values.append(
            [data['brief'][idx]['duration'], data['orb'][idx]['duration'], data['sift'][idx]['duration']])
        precision_values.append(
            [data['brief'][idx]['precision'], data['orb'][idx]['precision'], data['sift'][idx]['precision']])

    # Bar settings for the grouped bar chart
    bar_width = 0.25  # Width of each bar
    x = np.arange(len(indices))  # X locations for the bars

    # Create the plot with two subplots: one for duration, one for precision
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot bars for 'duration' in the first subplot
    for i, algorithm in enumerate(algorithms):
        ax1.bar(x + i * bar_width, [duration_values[idx][i] for idx in indices], bar_width, label=algorithm)

    # Plot bars for 'precision' in the second subplot
    for i, algorithm in enumerate(algorithms):
        ax2.bar(x + i * bar_width, [precision_values[idx][i] for idx in indices], bar_width, label=algorithm)

    # Customize the first subplot (Duration)
    ax1.set_xlabel('Index', fontsize=14)
    ax1.set_ylabel('Duration (seconds)', fontsize=14)
    ax1.set_title('Duration Comparison for Each Index', fontsize=16)
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels([f"{ALGORITHMS_ORDER[i]}" for i in indices], fontsize=12)
    ax1.legend()

    # Customize the second subplot (Precision)
    ax2.set_xlabel('Index', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title('Precision Comparison for Each Index', fontsize=16)
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels([f"{ALGORITHMS_ORDER[i]}" for i in indices], fontsize=12)
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the summary plot as an image
    output_path = './comparison_results_descriptors/comparison_summary.png'
    plt.savefig(output_path, format='png', dpi=300)

    if show_plot:
        # Optionally show the summary plot
        plt.show()


def compare_images():
    """
    Processes a set of images, compares the results of descriptors (ORB, SIFT, BRIEF), and saves the plots.

    This function will iterate over images in the 'images' directory and call compare_and_show_results for each.
    Afterward, it will save a summary comparison plot.
    """
    print("Starting to run descriptors experiment")
    dir_path = "images"
    images_names = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    for image in images_names:
        print(f"Starting process image: {image}")
        compare_and_show_results(f'./images/{image}', show_image=False)
    save_summary_run_plot()
