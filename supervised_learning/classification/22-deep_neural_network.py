#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    """Deep Neural Network performing binary classification"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            nodes = layers[l - 1]
            prev_nodes = nx if l == 1 else layers[l - 2]
            he_std = np.sqrt(2 / prev_nodes)
            self.__weights['W' + str(l)] = np.random.randn(nodes, prev_nodes) * he_std
            self.__weights['b' + str(l)] = np.zeros((nodes, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]
            A_prev = self.__cache['A' + str(l - 1)]
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(l)] = A
        return A, self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        L = self.__L
        weights_copy = self.__weights.copy()
        dZ = cache['A' + str(L)] - Y

        for l in reversed(range(1, L + 1)):
            A_prev = cache['A' + str(l - 1)]
            W = weights_copy['W' + str(l)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if l > 1:
                A_prev_l1 = cache['A' + str(l - 1)]
                W_prev = weights_copy['W' + str(l)]
                dZ = np.dot(W_prev.T, dZ) * (A_prev_l1 * (1 - A_prev_l1))
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
