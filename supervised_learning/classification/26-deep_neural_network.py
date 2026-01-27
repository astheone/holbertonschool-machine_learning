import pickle
import os

class DeepNeuralNetwork:
    # supozojmë që konstruktor dhe metoda të tjera ekzistojnë
    # def __init__(self, nx, layers):
    # def forward_prop(self, X):
    # def train(self, X, Y, iterations=5000, alpha=0.05, graph=True):
    # def evaluate(self, X, Y):

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
