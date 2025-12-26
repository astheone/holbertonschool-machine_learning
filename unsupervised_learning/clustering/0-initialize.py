#!/usr/bin/env python3
"""
Initialize K-means centroids
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    return np.random.uniform(min_vals, max_vals, size=(k, d))
