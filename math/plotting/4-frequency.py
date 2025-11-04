#!/usr/bin/env python3
"""
Module 4-frequency
This module contains a function that plots a histogram
of student grades for Project A.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades for Project A.
    The histogram has bins every 10 units and bars outlined in black.
    """
    plt.hist(student_grades, bins=10, range=(0, 100), edgecolor="k")
    plt.title("Project A")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.show
    plt.ylim(0, 30)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, step=10))