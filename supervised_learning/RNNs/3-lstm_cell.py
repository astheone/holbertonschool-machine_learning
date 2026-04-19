#!/usr/bin/env python3
"""Module that defines the LSTMCell class"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Class constructor
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        # Weights (Normal distribution)
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Biases (Zeros)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        x_t: numpy.ndarray of shape (m, i)
        h_prev: numpy.ndarray of shape (m, h)
        c_prev: numpy.ndarray of shape (m, h)
        Returns: h_next, c_next, y
        """
        # Concatenate x_t and h_prev for gate calculations
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))
        
        # Update gate
        u = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))
        
        # Intermediate cell state
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        
        # Next cell state
        c_next = f * c_prev + u * c_tilde
        
        # Output gate
        o = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))
        
        # Next hidden state
        h_next = o * np.tanh(c_next)
        
        # Output (y) using softmax
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, c_next, y
