#!/usr/bin/env python3
"""
Module 3-two
This module contains a function that plots the exponential decay
of two radioactive elements: C-14 and Ra-226.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots two line graphs showing the exponential decay
    of Carbon-14 and Radium-226.
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730  # Half-life of C-14
    t2 = 1600  # Half-life of Ra-226
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y1, 'r--', label="C-14")  # red dashed line
    plt.plot(x, y2, 'g-', label="Ra-226")  # solid green line

    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.show()
