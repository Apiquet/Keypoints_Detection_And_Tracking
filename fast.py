#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Features from Accelerated Segment Test (FAST) implementation
Algorithm to detect keypoints
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from matplotlib import pyplot as plt
from tqdm import tqdm


def detect(img, threshold=50, N=12):
    """
    Method to plot images with predicted and gt boxes

    Args:
        - (np.array) input image
        - (float) threshold to use to validate a pixel
    Return:
        - (np.array) vector of detected keypoints [Number of keypoints, x, y]
    """
    # NOT IMPLEMENTED
