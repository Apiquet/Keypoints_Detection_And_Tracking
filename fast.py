#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Features from Accelerated Segment Test (FAST) implementation
Algorithm to detect keypoints
"""

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from tqdm import tqdm

PIXELS_OF_INTEREST = {
    1:  np.array([ 0, -3]),
    5:  np.array([ 3,  0]),
    9:  np.array([ 0,  3]),
    13: np.array([-3,  0]),
    2:  np.array([ 1, -3]),
    3:  np.array([ 2, -2]),
    4:  np.array([ 3, -1]),
    6:  np.array([ 3,  1]),
    7:  np.array([ 2,  2]),
    8:  np.array([ 1,  3]),
    10: np.array([-1,  3]),
    11: np.array([-2,  2]),
    12: np.array([-3,  1]),
    14: np.array([-3,  1]),
    15: np.array([-2,  2]),
    16: np.array([-1,  3])
}

def get_pixel_value(array, pixel_position):
    """
    Function to get a value from a numpy array

    Args:
        - (np.array) input image
        - (np.array) index [x, y]
    Return:
        - (int) value
    """
    return array[pixel_position[0], pixel_position[1]]


def detect_with_adaptive_threshold(
        img, nb_keypoints, N=12, step=3, epsilon=50,
        percentage=0.1, init_threshold=None):
    """
    Function to detect keypoints with adaptive threshold

    Args:
        - (np.array) input image
        - (int) number of wanted keypoints
        - (int) min number of neighbor to validate a pixel
        - (int) step to do 1/step pixels
            (if step=1: computation done on every pixel)
        - (int) epsilon to accept a number of keypoints
        - (float) percentage to change the threshold per iteration
        - (int) value to initialize the threshold (default is 15)
    Return:
        - (np.array) vector of detected keypoints
            [Number of keypoints, x, y]
    """
    # create threshold on first call
    if not hasattr(detect_with_adaptive_threshold, "threshold") or\
            init_threshold is not None:
        detect_with_adaptive_threshold.threshold = 15

    # use detect function to get the keypoints
    keypoints = detect(img, detect_with_adaptive_threshold.threshold,
                       N=N, step=step)
    
    # adapt the threshold in function of the number of keypoints
    if keypoints.shape[0] > nb_keypoints + epsilon:
        change = detect_with_adaptive_threshold.threshold*percentage
        if change < 1:
            change = 1
        else:
            change = math.floor(change)
        detect_with_adaptive_threshold.threshold += change
    elif keypoints.shape[0] < nb_keypoints - epsilon:
        change = detect_with_adaptive_threshold.threshold*percentage
        if change < 1:
            change = 1
        else:
            change = math.floor(change)
        detect_with_adaptive_threshold.threshold -= change

    return keypoints


def detect(img, threshold=15, N=12, step=3):
    """
    Function to detect keypoints on an image

    Args:
        - (np.array) input image
        - (int) threshold to use to validate a pixel
        - (int) min number of neighbor to validate a pixel
        - (int) step to do 1/step pixels
            (if step=1: computation done on every pixel)
    Return:
        - (np.array) vector of detected keypoints
            [Number of keypoints, x, y]
    """
    final_keypoints = []

    # loop on pixels with a step
    for y in range(3, img.shape[1]-3, step):
        for x in range(3, img.shape[0]-3, step):
            neighbors_validated = 0
            pixel_position = np.array([x, y])
            
            # get pixel value
            pixel_value = get_pixel_value(img, pixel_position)
            
            # calculate bounds to validate a neighboring pixel
            lower_bound = pixel_value - threshold
            higher_bound = pixel_value + threshold
            
            for i, (key, value) in enumerate(
                    PIXELS_OF_INTEREST.items()):
                # get a neighboring pixel value
                neighbor_pixel_value = get_pixel_value(
                    img, pixel_position+value)
                # verify criterion
                if neighbor_pixel_value <= lower_bound or\
                        neighbor_pixel_value >= higher_bound:
                    neighbors_validated += 1
                # the first 4 pixels are 1, 5, 9, 13
                # if less than 3 of them are validated
                # invalidate the current pixel as keypoints
                if i == 3 and neighbors_validated < 3:
                    break
            
            if neighbors_validated >= N:
                final_keypoints.append(pixel_position)
    return np.asarray(final_keypoints)


def draw(frame, keypoints):
    """
    Function to draw keypoints on an image

    Args:
        - (cv2.image) input image
        - (np.array) vector of detected keypoints [Number of keypoints, x, y]
    Return:
        - (cv2.image) input image with drawn keypoints
    """
    img = frame.copy()
    for point in keypoints:
        cv2.circle(img, (point[1], point[0]), 1, (0, 255, 255), 4)
    return img
