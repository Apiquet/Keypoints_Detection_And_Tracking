#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Features from Accelerated Segment Test (FAST) implementation
Algorithm to detect keypoints
"""

import math
from typing import Tuple

import cv2
import numpy as np

PIXELS_OF_INTEREST = {
    1: np.array([0, -3]),
    5: np.array([3, 0]),
    9: np.array([0, 3]),
    13: np.array([-3, 0]),
    2: np.array([1, -3]),
    3: np.array([2, -2]),
    4: np.array([3, -1]),
    6: np.array([3, 1]),
    7: np.array([2, 2]),
    8: np.array([1, 3]),
    10: np.array([-1, 3]),
    11: np.array([-2, 2]),
    12: np.array([-3, 1]),
    14: np.array([-3, 1]),
    15: np.array([-2, 2]),
    16: np.array([-1, 3]),
}


def get_pixel_value(array: np.array, pixel_position: Tuple[int, int]) -> int:
    """
    Function to get a value from a numpy array

    Args:
        - array: input image
        - pixel_position: index [x, y]
    Return:
        - array value at coordinates
    """
    return array[pixel_position[0], pixel_position[1]]


def extract_patches(array: np.array, rows: int, cols: int) -> np.array:
    """
    Divide a 2D array into patches of size [rows*cols]

    Args:
        - array: input array to divide
        - rows: number of rows in a patch
        - cols: number of cols in a patch
    Return:
        - array of size [nb patch, rows, cols]
    """
    patches = (
        array.reshape(array.shape[0] // rows, rows, -1, cols).swapaxes(1, 2).reshape(-1, rows, cols)
    )
    return patches


def detect_with_adaptive_threshold_and_grid(
    img: np.array,
    nb_keypoints: int,
    N: int = 12,
    step: int = 5,
    epsilon: int = 50,
    percentage: float = 0.1,
    init_thresholds: int = None,
    cols: int = 4,
    rows: int = 3,
) -> np.array:
    """
    Function to detect keypoints with adaptive threshold on multiple cells

    Args:
        - img: input image
        - nb_keypoints: number of wanted keypoints
        - N: min number of neighbor to validate a pixel
        - step: step to do 1/step pixels
            (if step=1: computation done on every pixel)
        - epsilon: epsilon to accept a number of keypoints
        - percentage: percentage to change the threshold per iteration
        - init_threshold: value to initialize the threshold (default is 15)
        - cols: number of columns
        - rows: number of rows
    Return:
        - array of detected keypoints
            [Number of keypoints, x, y]
    """
    # create threshold on first call
    nb_cells = cols * rows
    epsilon = epsilon / nb_cells
    if not hasattr(detect_with_adaptive_threshold_and_grid, "thresholds"):
        detect_with_adaptive_threshold_and_grid.thresholds = [15 for _ in range(nb_cells)]
    if init_thresholds is not None:
        if len(init_thresholds) != cols * rows:
            print("Init thresholds should be a list of size cols*rows")
            return
        detect_with_adaptive_threshold_and_grid.thresholds = init_thresholds

    # calculate number of rows and cols per cell
    nb_cols_per_cell = img.shape[1] // cols
    nb_rows_per_cell = img.shape[0] // rows
    # calculate number of keypoints wanted per cell
    nb_keypoints_per_cell = nb_keypoints // nb_cells
    # divide image by cols*rows cells
    detect_with_adaptive_threshold_and_grid.patches = extract_patches(
        img, nb_rows_per_cell, nb_cols_per_cell
    )

    detect_with_adaptive_threshold_and_grid.nb_keypoints_per_cell = []
    detect_with_adaptive_threshold_and_grid.keypoints_per_cell = []

    for i, patch in enumerate(detect_with_adaptive_threshold_and_grid.patches):
        # use detect function to get the keypoints
        keypoints = detect(
            patch, detect_with_adaptive_threshold_and_grid.thresholds[i], N=N, step=step
        )

        nb_keypoints_in_cell = keypoints.shape[0]
        detect_with_adaptive_threshold_and_grid.nb_keypoints_per_cell.append(nb_keypoints_in_cell)
        detect_with_adaptive_threshold_and_grid.keypoints_per_cell.append(keypoints)
        # adapt the threshold in function of the number of keypoints
        if nb_keypoints_in_cell > nb_keypoints_per_cell + epsilon:
            change = detect_with_adaptive_threshold_and_grid.thresholds[i] * percentage
            if change < 1:
                change = 1
            else:
                change = math.floor(change)
            detect_with_adaptive_threshold_and_grid.thresholds[i] += change
        elif nb_keypoints_in_cell < nb_keypoints_per_cell - epsilon:
            change = detect_with_adaptive_threshold_and_grid.thresholds[i] * percentage
            if change < 1:
                change = 1
            else:
                change = math.floor(change)
            detect_with_adaptive_threshold_and_grid.thresholds[i] -= change

        # convert patch number into position in the image
        patch_x_pos = i // cols
        patch_y_pos = i - patch_x_pos * cols
        # offset to keypoints to convert patch position to image position
        offset_pos = np.array([patch_x_pos * nb_rows_per_cell, patch_y_pos * nb_cols_per_cell])
        keypoints = np.array([kp + offset_pos for kp in keypoints])

        if 'keypoints_per_cell' not in locals():
            keypoints_per_cell = keypoints
        elif len(keypoints) != 0:
            keypoints_per_cell = np.concatenate([keypoints_per_cell, keypoints])

    return keypoints_per_cell


def detect_with_adaptive_threshold(
    img: np.array,
    nb_keypoints: int,
    N: int = 12,
    step: int = 3,
    epsilon: int = 50,
    percentage: float = 0.1,
    init_threshold: int = None,
) -> np.array:
    """
    Function to detect keypoints with adaptive threshold

    Args:
        - img: input image
        - nb_keypoints: number of wanted keypoints
        - N: min number of neighbor to validate a pixel
        - step: step to do 1/step pixels
            (if step=1: computation done on every pixel)
        - epsilon: epsilon to accept a number of keypoints
        - percentage: percentage to change the threshold per iteration
        - init_threshold: value to initialize the threshold (default is 15)
    Return:
        - array of detected keypoints
            [Number of keypoints, x, y]
    """
    # create threshold on first call
    if not hasattr(detect_with_adaptive_threshold, "threshold"):
        detect_with_adaptive_threshold.threshold = 15
    if init_threshold is not None:
        detect_with_adaptive_threshold.threshold = init_threshold

    # use detect function to get the keypoints
    keypoints = detect(img, detect_with_adaptive_threshold.threshold, N=N, step=step)

    # adapt the threshold in function of the number of keypoints
    if keypoints.shape[0] > nb_keypoints + epsilon:
        change = detect_with_adaptive_threshold.threshold * percentage
        if change < 1:
            change = 1
        else:
            change = math.floor(change)
        detect_with_adaptive_threshold.threshold += change
    elif keypoints.shape[0] < nb_keypoints - epsilon:
        change = detect_with_adaptive_threshold.threshold * percentage
        if change < 1:
            change = 1
        else:
            change = math.floor(change)
        detect_with_adaptive_threshold.threshold -= change

    return keypoints


def detect(img: np.array, threshold: int = 15, N: int = 12, step: int = 3) -> np.array:
    """
    Function to detect keypoints on an image

    Args:
        - img: input image
        - threshold: threshold to use to validate a pixel
        - N: min number of neighbor to validate a pixel
        - step: step to do 1/step pixels
            (if step=1: computation done on every pixel)
    Return:
        - array of detected keypoints
            [Number of keypoints, x, y]
    """
    final_keypoints = []

    # loop on pixels with a step
    for y in range(3, img.shape[1] - 3, step):
        for x in range(3, img.shape[0] - 3, step):
            neighbors_validated = 0
            pixel_position = np.array([x, y])

            # get pixel value
            pixel_value = get_pixel_value(img, pixel_position)

            # calculate bounds to validate a neighboring pixel
            lower_bound = pixel_value - threshold
            higher_bound = pixel_value + threshold

            for i, (key, value) in enumerate(PIXELS_OF_INTEREST.items()):
                # get a neighboring pixel value
                neighbor_pixel_value = get_pixel_value(img, pixel_position + value)
                # verify criterion
                if neighbor_pixel_value <= lower_bound or neighbor_pixel_value >= higher_bound:
                    neighbors_validated += 1
                # the first 4 pixels are 1, 5, 9, 13
                # if less than 3 of them are validated
                # invalidate the current pixel as keypoints
                if i == 3 and neighbors_validated < 3:
                    break

            if neighbors_validated >= N:
                final_keypoints.append(pixel_position)
    return np.asarray(final_keypoints)


def draw(frame: np.ndarray, keypoints: np.array) -> np.ndarray:
    """
    Function to draw keypoints on an image

    Args:
        - frame: input image
        - keypoints: vector of detected keypoints [Number of keypoints, x, y]
    Return:
        - input image with drawn keypoints
    """
    img = frame.copy()
    for point in keypoints:
        cv2.circle(img, (point[1], point[0]), 1, (0, 255, 255), 4)
    return img
