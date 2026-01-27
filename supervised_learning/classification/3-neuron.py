#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""

import numpy as np


class Neuron:
    """
    Defines a neuron
    """

    def __init__(self, nx):
        """
        Initializes the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation
        Updates __A
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        Calculates cost using logistic regression
        Y: correct labels (1, m)
        A: activated output (1, m)
        Returns: cost (scalar)
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
