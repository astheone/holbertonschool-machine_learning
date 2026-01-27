#!/usr/bin/env python3
"""Function that converts a one-hot matrix into a vector of labels."""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded matrix into a vector of numeric labels.

    Parameters:
    - one_hot: numpy.ndarray of shape (classes, m) containing one-hot encoding

    Returns:
    - numpy.ndarray of shape (m,) containing the numeric labels for each example
      or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    try:
        # argmax returns the index of the max value in each column
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
