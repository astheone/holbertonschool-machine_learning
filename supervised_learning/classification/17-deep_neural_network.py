#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Class constructor
        nx: number of input features
        layers: list of number of nodes in each layer
        """

        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        # Private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases using He initialization
        for l in range(self.__L):
            layer_size = layers[l]
            prev_size = nx if l == 0 else layers[l - 1]

            self.__weights[f"W{l + 1}"] = (
                np.random.randn(layer_size, prev_size)
                * np.sqrt(2 / prev_size)
            )
            self.__weights[f"b{l + 1}"] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Returns the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Returns the cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Returns the weights dictionary"""
        return self.__weights
