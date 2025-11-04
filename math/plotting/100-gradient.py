#!/usr/bin/env python3
"""
Module 100-gradient
This module contains a function that plots a scatter plot
showing sampled elevations on a mountain.
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Creates a scatter plot of sampled elevations on a mountain.
    The color of each point represents its elevation.
    """
    np.random.seed(5)
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    plt.figure(figsize=(6.4, 4.8))
    scatter = plt.scatter(x, y, c=z, cmap='terrain', s=20)
    cbar = plt.colorbar(scatter)
    cbar.set_label("elevation (m)")
    plt.title("Mountain Elevation")
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    plt.show()
