#!/usr/bin/env python3
"""Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights using gradient descent with Dropout"""
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            D = cache['D' + str(i - 1)]
            dA = np.matmul(W.T, dZ)
            dA = dA * D / keep_prob
            dZ = dA * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
