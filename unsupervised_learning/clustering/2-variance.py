#!/usr/bin/env python3
"""
Variance module
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset

    Arguments:
        X -- numpy.ndarray of shape (n, d)
        C -- numpy.ndarray of shape (k, d)

    Returns:
        var -- total variance (float)
    """
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    # Compute distances between each point and each centroid
    distances = np.linalg.norm(X[:, None] - C[None, :], axis=2)
    # Assign each point to closest centroid
    clss = np.argmin(distances, axis=1)
    # Compute variance (sum of squared distances)
    diff = X - C[clss]
    var = np.sum(np.square(diff))

    return var
