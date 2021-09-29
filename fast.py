#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Features from Accelerated Segment Test (FAST) implementation
Algorithm to detect keypoints
"""

import cv2
import numpy as np
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

def get_pixel_value(img, pixel_position):
    """
    Method to plot images with predicted and gt boxes

    Args:
        - (np.array) input image
        - (np.array) pixel position [x, y]
    Return:
        - (int) pixel value
    """
    return img[pixel_position[0], pixel_position[1]]


def detect(img, threshold=50, N=12, step=3):
    """
    Method to plot images with predicted and gt boxes

    Args:
        - (np.array) input image
        - (float) threshold to use to validate a pixel
    Return:
        - (np.array) vector of detected keypoints [Number of keypoints, x, y]
    """
    final_keypoint = []
    for y in range(3, img.shape[1]-3, step):
        for x in range(3, img.shape[0]-3, step):
            neighbors_validated = 0
            pixel_position = np.array([x, y])
            pixel_value = get_pixel_value(img, pixel_position)
            lower_bound = pixel_value - threshold
            higher_bound = pixel_value + threshold
            
            for i, (key, value) in enumerate(PIXELS_OF_INTEREST.items()):
                neighbor_pixel_value = get_pixel_value(img, pixel_position+value)
                if neighbor_pixel_value <= lower_bound or neighbor_pixel_value >= higher_bound:
                    neighbors_validated += 1
                if i == 3 and neighbors_validated < 3:
                    break
            
            if neighbors_validated >= N:
                final_keypoint.append(pixel_position)
    return np.asarray(final_keypoint)


def draw(img, keypoints):
    """
    Method to plot images with predicted and gt boxes

    Args:
        - (cv2.image) input image
        - (np.array) vector of detected keypoints [Number of keypoints, x, y]
    Return:
        - (cv2.image) input image with drawn keypoints
    """
    for point in keypoints:
        cv2.circle(img, (point[1], point[0]), 1, (0, 255, 255), 4)
    return img
