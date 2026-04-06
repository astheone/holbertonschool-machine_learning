#!/usr/bin/env python3
"""Convolutional Back Propagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            a_slice = A_prev_pad[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(c_new):
                dA_prev_pad[
                    :, i*sh:i*sh+kh, j*sw:j*sw+kw, :
                ] += W[:, :, :, k] * dZ[:, i, j, k][:, None, None, None]
                dW[:, :, :, k] += np.sum(
                    a_slice * dZ[:, i, j, k][:, None, None, None],
                    axis=0
                )

    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
