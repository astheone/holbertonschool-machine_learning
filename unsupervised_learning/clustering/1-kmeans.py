#!/usr/bin/env python3
"""
K-means clustering
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n
