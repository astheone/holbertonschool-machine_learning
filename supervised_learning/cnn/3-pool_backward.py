#!/usr/bin/env python3
"""Pooling Back Propagation"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer"""
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            patch = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                mask = (patch == np.max(patch, axis=(1, 2), keepdims=True))
                dA_prev[
                    :, i*sh:i*sh+kh, j*sw:j*sw+kw, :
                ] += mask * dA[:, i, j, :][:, None, None, :]
            else:
                avg = dA[:, i, j, :][:, None, None, :] / (kh * kw)
                dA_prev[
                    :, i*sh:i*sh+kh, j*sw:j*sw+kw, :
                ] += np.ones((m, kh, kw, c)) * avg
    return dA_prev
