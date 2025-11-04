#!/usr/bin/env python3
"""
Module 0-line
This module contains a function that plots y = x³ as a red line graph.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots a line graph of y = x³ for x values from 0 to 10.
    The line is displayed as a solid red line.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(np.arange(0, 11), y, 'r-')
    plt.xlim(0, 10)
    plt.show()
