#!/usr/bin/env python3
import tensorflow.keras as K

def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Args:
        network (K.models.Model): The network model to test.
        data (numpy.ndarray): The input data to test the model with.
        labels (numpy.ndarray): The correct one-hot labels of the data.
        verbose (bool, optional): Determines if output should be printed during the testing process. Defaults to True.

    Returns:
        tuple: The loss and accuracy of the model with the testing data, respectively.
    """
    # Test the model on the input data and labels
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)

    return loss, accuracy
