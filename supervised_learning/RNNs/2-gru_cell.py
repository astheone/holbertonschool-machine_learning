#!/usr/bin/env python3
"""Module that contains the GRUCell class"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        # Weights and biases for update gate (z)
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        # Weights and biases for reset gate (r)
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        # Weights and biases for intermediate hidden state (h tilde)
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        # Weights and biases for output (y)
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        x_t: (m, i) input data
        h_prev: (m, h) previous hidden state
        Returns: h_next, y
        """
        # Concatenate x_t and h_prev for gate calculations
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Update Gate (z_t)
        z_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wz) + self.bz)))

        # Reset Gate (r_t)
        r_t = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wr) + self.br)))

        # Intermediate Hidden State (h_tilde)
        # Apply reset gate to h_prev before concatenation
        h_prev_reset = r_t * h_prev
        concat_h_tilde = np.concatenate((h_prev_reset, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_h_tilde, self.Wh) + self.bh)

        # Next Hidden State (h_next)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output (y) using softmax
        y_linear = np.dot(h_next, self.Wy) + self.by
        y_exp = np.exp(y_linear)
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, y
