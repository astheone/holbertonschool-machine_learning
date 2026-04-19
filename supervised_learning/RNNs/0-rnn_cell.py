#!/usr/bin/env python3
"""Module that contains the class RNNCell"""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        x_t: (m, i) input data
        h_prev: (m, h) previous hidden state
        Returns: h_next, y
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Wh) + self.bh)
        z = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        return h_next, y
