#!/usr/bin/env python3
"""
This module contains a function to decode a one-hot encoded matrix
back into a numeric label vector.
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded matrix into a vector of numeric labels.

    one_hot: numpy.ndarray of shape (classes, m) representing one-hot encoded labels

    Returns: numpy.ndarray of shape (m,) containing numeric labels,
             or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.ndim != 2:
        return None

    # argmax along the rows gives the original class labels
    return np.argmax(one_hot, axis=0)
