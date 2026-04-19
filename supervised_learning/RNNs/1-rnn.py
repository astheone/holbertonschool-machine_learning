#!/usr/bin/env python3
"""Module that contains the rnn function"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    rnn_cell: instance of RNNCell
    X: numpy.ndarray of shape (t, m, i)
    h_0: numpy.ndarray of shape (m, h)
    Returns: H, Y
    """
    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.by.shape[1]

    # Initialize H with shape (t + 1, m, h)
    # H[0] is h_0, so we need +1 for the time dimension
    H = np.zeros((t + 1, m, h))
    # Initialize Y with shape (t, m, o)
    Y = np.zeros((t, m, o))

    # Set the initial hidden state
    H[0] = h_0
    h_prev = h_0

    # Iterate through time steps
    for step in range(t):
        # rnn_cell.forward takes h_prev and x_t
        h_next, y = rnn_cell.forward(h_prev, X[step])

        # Store results
        H[step + 1] = h_next
        Y[step] = y

        # Update h_prev for the next iteration
        h_prev = h_next

    return H, Y
