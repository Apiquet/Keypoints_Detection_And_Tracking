#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Features from Accelerated Segment Test (FAST) implementation
Algorithm to detect keypoints
"""

import cv2
from PIL import Image
import numpy as np
import math
from matplotlib import pyplot as plt
from tqdm import tqdm


def compute():
    return 0


def draw(frame, keypoints, flow, draw_kp_with_zero_flow=False, factor=1):
    """
    Function to draw keypoints on an image with optical flow

    Args:
        - (cv2.image) input image
        - (np.array) vector of detected keypoints [Number of keypoints, x, y]
        - (np.array) flow for each keypoints [Number of keypoints, x, y]
    Return:
        - (cv2.image) input image with drawn keypoints and flow
    """
    img = frame.copy()
    for point in keypoints:
        x_flow = int(flow[1][point[0], point[1]]*factor)
        y_flow = int(flow[0][point[0], point[1]]*factor)
        if x_flow == 0 and y_flow == 0 and not draw_kp_with_zero_flow:
            continue
        end_flow = (point[1]-x_flow, point[0]-y_flow)
        cv2.circle(img, (point[1], point[0]), 2, (0, 255, 255), 4)
        cv2.line(img, (point[1], point[0]), end_flow, (255, 0, 0),
                 thickness=3)
    return img
