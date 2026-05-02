"""
Statistical Anomaly Detection with Images

This script uses confidence intervals to detect anomalous images based on
average pixel intensity values across an 8x8 grid.

Expected folder structure:

sample_images/
    non-anomalous_images/
        image1.jpeg
        image2.jpeg
        ...
    anomalous_images/
        image1.jpeg
        image2.jpeg
        ...

Run:
    python anomaly_detection.py
"""

from os import listdir
from os.path import join

import cv2
import numpy as np
import pandas as pd
from scipy.stats import t


GRID_SIZE = 8
CONFIDENCE_LEVEL = 0.97


def load_image_paths(folder_path):
    """Return a list of JPEG image paths from a folder."""
    return [
        join(folder_path, file_name)
        for file_name in listdir(folder_path)
        if file_name.lower().endswith(".jpeg")
    ]


def load_grayscale_images(image_paths):
    """Load images as grayscale arrays."""
    images = []

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            images.append(image)

    return images


def compute_average_pixel_intensities(image, grid_size=GRID_SIZE):
    """
    Split one grayscale image into a grid and calculate the average
    pixel intensity inside each grid cell.
    """
    height, width = image.shape
    cell_height = height // grid_size
    cell_width = width // grid_size

    mean_per_cell = np.zeros((grid_size, grid_size))

    for row in range(grid_size):
        for column in range(grid_size):
            y1 = row * cell_height
            y2 = (row + 1) * cell_height
            x1 = column * cell_width
            x2 = (column + 1) * cell_width

            cell = image[y1:y2, x1:x2]
            mean_per_cell[row, column] = np.mean(cell)

    return mean_per_cell


def compute_all_cell_averages(images, grid_size=GRID_SIZE):
    """
    Convert a list of images into a 3D array.

    Output shape:
        number of images x grid rows x grid columns
    """
    averages = []

    for image in images:
        cell_averages = compute_average_pixel_intensities(image, grid_size)
        averages.append(cell_averages)

    return np.array(averages)


def build_cell_intervals(normal_averages, confidence_level=CONFIDENCE_LEVEL, use_standard_error=False):
    """
    Build a confidence interval for each grid cell using the normal images.

    If use_standard_error is True, the interval is based on standard error.
    If False, the interval is based on sample standard deviation, which creates
    wider intervals for classifying individual image values.
    """
    alpha = 1 - confidence_level
    grid_size = normal_averages.shape[1]

    intervals = pd.DataFrame(columns=["Row", "Col", "CI_Lower", "CI_Upper"])

    for row in range(grid_size):
        for column in range(grid_size):
            samples = normal_averages[:, row, column]

            sample_size = len(samples)
            sample_mean = np.mean(samples)
            sample_std = np.std(samples, ddof=1)

            t_critical = t.ppf(1 - alpha / 2, df=sample_size - 1)

            if use_standard_error:
                spread = sample_std / np.sqrt(sample_size)
            else:
                spread = sample_std

            margin_of_error = t_critical * spread
            lower_bound = sample_mean - margin_of_error
            upper_bound = sample_mean + margin_of_error

            intervals.loc[len(intervals)] = [row, column, lower_bound, upper_bound]

    return intervals


def classify_image(cell_averages, intervals):
    """
    Classify one image as anomalous if any grid cell falls outside
    the expected interval for that cell.
    """
    grid_size = cell_averages.shape[0]

    for row in range(grid_size):
        for column in range(grid_size):
            cell_value = cell_averages[row, column]

            interval = intervals[
                (intervals["Row"] == row) &
                (intervals["Col"] == column)
            ]

            lower_bound = interval["CI_Lower"].values[0]
            upper_bound = interval["CI_Upper"].values[0]

            if cell_value < lower_bound or cell_value > upper_bound:
                return True

    return False


def evaluate_model(anomalous_averages, normal_averages, intervals):
    """
    Calculate true positive rate and false positive rate.

    True positive rate:
        Percent of anomalous images correctly flagged as anomalous.

    False positive rate:
        Percent of normal images incorrectly flagged as anomalous.
    """
    true_positives = 0
    false_positives = 0

    for image_averages in anomalous_averages:
        if classify_image(image_averages, intervals):
            true_positives += 1

    for image_averages in normal_averages:
        if classify_image(image_averages, intervals):
            false_positives += 1

    true_positive_rate = true_positives / len(anomalous_averages)
    false_positive_rate = false_positives / len(normal_averages)

    return true_positive_rate, false_positive_rate


def main():
    normal_folder = "./sample_images/non-anomalous_images/"
    anomalous_folder = "./sample_images/anomalous_images/"

    normal_paths = load_image_paths(normal_folder)
    anomalous_paths = load_image_paths(anomalous_folder)

    normal_images = load_grayscale_images(normal_paths)
    anomalous_images = load_grayscale_images(anomalous_paths)

    normal_averages = compute_all_cell_averages(normal_images)
    anomalous_averages = compute_all_cell_averages(anomalous_images)

    intervals = build_cell_intervals(
        normal_averages,
        confidence_level=CONFIDENCE_LEVEL,
        use_standard_error=False
    )

    true_positive_rate, false_positive_rate = evaluate_model(
        anomalous_averages,
        normal_averages,
        intervals
    )

    print(f"True Positive Rate: {true_positive_rate:.2f}")
    print(f"False Positive Rate: {false_positive_rate:.2f}")


if __name__ == "__main__":
    main()
