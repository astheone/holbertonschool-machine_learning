#!/usr/bin/env python3
"""Pooling Forward Propagation"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h_prev - kh) // sh + 1
    ow = (w_prev - kw) // sw + 1
    output = np.zeros((m, oh, ow, c_prev))

    for i in range(oh):
        for j in range(ow):
            patch = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))
    return output
