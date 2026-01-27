#!/usr/bin/env python3
import pickle
import os


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """Constructor supozojmë se ekziston."""
        pass

    def forward_prop(self, X):
        """Supozim."""
        pass

    def train(self, X, Y, iterations=5000, alpha=0.05, graph=True):
        """Supozim."""
        pass

    def evaluate(self, X, Y):
        """Supozim."""
        pass

    def save(self, filename):
        """Saves the instance object to a file in pickle format."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object."""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
