#!/usr/bin/env python3
"""Test a neural network model"""

def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    Args:
        network: trained model to test
        data: input data
        labels: correct one-hot labels
        verbose: print progress if True

    Returns:
        loss, accuracy
    """
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return loss, accuracy
