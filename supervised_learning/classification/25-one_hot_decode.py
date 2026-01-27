#!/usr/bin/env python3
"""Function that converts a one-hot matrix into a vector of labels."""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded numpy.ndarray into a vector of labels.

    Parameters:
    - one_hot: numpy.ndarray of shape (classes, m) with one-hot encoding

    Returns:
    - numpy.ndarray of shape (m,) with numeric labels for each example,
      or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.size == 0:
        return None

    decoded = np.argmax(one_hot, axis=0)
    return decoded
