#!/usr/bin/env python3
"""
K-means clustering
"""
import numpy as np
initialize = __import__('0-initialize').initialize


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

    n, d = X.shape

    # MUST use initialize from task 0
    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    for _ in range(iterations):
        C_prev = C.copy()

        # Assign clusters
        distances = np.linalg.norm(X[:, None] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):
            points = X[clss == i]
            if points.size == 0:
                C[i] = np.random.uniform(min_vals, max_vals, d)
            else:
                C[i] = points.mean(axis=0)

        if np.allclose(C, C_prev):
            break

    return C, clss
