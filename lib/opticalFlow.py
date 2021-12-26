#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lucas-Kanade re implementation
Algorithm to track key points
"""

import cv2
import numpy as np


def compute():
    return 0


def draw(
    frame: np.ndarray,
    keypoints: np.array,
    flow: np.array,
    draw_kp_with_zero_flow: bool = False,
    factor: int = 1,
) -> np.ndarray:
    """
    Function to draw keypoints on an image with optical flow

    Args:
        - frame: input image
        - keypoints: vector of detected keypoints [Number of keypoints, x, y]
        - flow: flow for each keypoints [Number of keypoints, x, y]
        - draw_kp_with_zero_flow: to draw kp with no movement
    Return:
        - input image with drawn keypoints and flow
    """
    img = frame.copy()
    w, h, c = frame.shape
    new_keypoints_pos = []
    for point in keypoints:
        x_flow = int(flow[1][point[0], point[1]] * factor)
        y_flow = int(flow[0][point[0], point[1]] * factor)

        x_end_flow = min(w - 1, max(0, point[0] + x_flow))
        y_end_flow = min(h - 1, max(0, point[1] + y_flow))

        end_flow = np.array([x_end_flow, y_end_flow])
        new_keypoints_pos.append(end_flow)

        if x_flow == 0 and y_flow == 0 and not draw_kp_with_zero_flow:
            continue
        cv2.line(img, (point[1], point[0]), (end_flow[1], end_flow[0]), (255, 0, 0), thickness=3)
        cv2.circle(img, (end_flow[1], end_flow[0]), 2, (0, 255, 255), 4)
    return img, np.asarray(new_keypoints_pos)
