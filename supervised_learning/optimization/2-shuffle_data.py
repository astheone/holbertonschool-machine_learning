#!/usr/bin/env python3
"""Module for shuffling data."""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X: first numpy.ndarray of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features in X
        Y: second numpy.ndarray of shape (m, ny) to shuffle
            m is the same number of data points as in X
            ny is the number of features in Y

    Returns:
        The shuffled X and Y matrices
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    
    return X_shuffled, Y_shuffled
```

**Shpjegim:**

1. **`np.random.permutation(m)`**: Krijon një array të permutuar të indekseve `[0, 1, 2, ..., m-1]`
   - Për shembull me `m=5`: `[2, 0, 1, 3, 4]`

2. **`X[permutation]`**: Përdor permutacionin për të ri-renditur rreshtrat e X

3. **E njëjta permutacion për X dhe Y**: Kjo siguron që çdo data point dhe label-i korrespondues qëndrojnë të lidhur

**Shembull:**
```
Origjinale:
X: [[1,2], [3,4], [5,6]]
Y: [[11,12], [13,14], [15,16]]

permutation: [2, 0, 1]

Shuffled:
X: [[5,6], [1,2], [3,4]]
Y: [[15,16], [11,12], [13,14]]
