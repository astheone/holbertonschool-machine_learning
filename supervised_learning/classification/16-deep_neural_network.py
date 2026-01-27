#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        # Validimi i nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validimi i layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        # Inicializimi i atributëve publikë
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Vetëm një loop për peshat dhe bias
        for l in range(1, self.L + 1):
            nodes = layers[l - 1]
            prev_nodes = nx if l == 1 else layers[l - 2]
            he_std = np.sqrt(2 / prev_nodes)
            self.weights['W' + str(l)] = np.random.randn(nodes, prev_nodes) * he_std
            self.weights['b' + str(l)] = np.zeros((nodes, 1))
