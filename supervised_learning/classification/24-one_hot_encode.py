#!/usr/bin/env python3
"""
This module contains a function for one-hot encoding numeric labels
into a binary matrix suitable for machine learning tasks.
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Y: numpy.ndarray of shape (m,) containing numeric class labels
    classes: maximum number of classes

    Returns: one-hot encoded matrix of shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if Y.ndim != 1 or classes < 2 or classes <= np.max(Y):
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot
