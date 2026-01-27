#!/usr/bin/env python3
"""Function that converts a one-hot matrix into a vector of labels."""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded numpy.ndarray into a vector of labels.

    Parameters:
    - one_hot: numpy.ndarray of shape (classes, m)

    Returns:
    - numpy.ndarray of shape (m,) with numeric labels
      or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    return np.argmax(one_hot, axis=0)
