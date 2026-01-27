#!/usr/bin/env python3
"""Function that converts a numeric label vector into a one-hot matrix."""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot encoded matrix.

    Parameters:
    - Y: numpy.ndarray of shape (m,) with numeric class labels
    - classes: int, maximum number of classes

    Returns:
    - numpy.ndarray of shape (classes, m) with one-hot encoding
      or None if input is invalid
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes < 2 or np.max(Y) >= classes:
        return None
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot
