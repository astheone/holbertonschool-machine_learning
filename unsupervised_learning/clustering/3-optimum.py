#!/usr/bin/env python3
"""
3-optimum.py
Determine the optimum number of clusters using variance
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance

def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    X: numpy.ndarray of shape (n, d)
    kmin: minimum number of clusters to check
    kmax: maximum number of clusters to check
    iterations: max iterations for kmeans

    Returns: results, d_vars
    """
    if type(X) is not np.ndarray or X.size == 0:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    if type(kmin) is not int or kmin < 1 or kmin >= kmax:
        return None, None

    results = []
    d_vars = []

    # Loop through cluster numbers
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
    
    # Compute variance differences
    base_var = variance(X, results[0][0])
    for i, (C, _) in enumerate(results):
        d_vars.append(variance(X, C) - base_var)

    return results, np.array(d_vars)
