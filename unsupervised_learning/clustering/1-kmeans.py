#!/usr/bin/env python3
"""
K-means clustering module
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means using uniform distribution.
    X: numpy.ndarray of shape (n, d)
    k: number of clusters
    Returns: numpy.ndarray of shape (k, d)
    """
    try:
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        return np.random.uniform(min_vals, max_vals, (k, X.shape[1]))
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering
    X: numpy.ndarray of shape (n, d)
    k: number of clusters
    iterations: max number of iterations
    Returns: C (centroids), clss (cluster indices)
    """
    try:
        n, d = X.shape
        C = initialize(X, k)
        clss = np.zeros(n, dtype=int)

        for _ in range(iterations):
            # Compute distances and assign clusters
            distances = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
            new_clss = np.argmin(distances, axis=1)

            if np.array_equal(clss, new_clss):
                # Early stopping if clusters don't change
                break
            clss = new_clss

            # Update centroids
            for i in range(k):
                points = X[clss == i]
                if len(points) == 0:
                    # Reinitialize empty cluster
                    C[i] = np.random.uniform(X.min(axis=0), X.max(axis=0), d)
                else:
                    C[i] = points.mean(axis=0)

        return C, clss
    except Exception:
        return None, None
