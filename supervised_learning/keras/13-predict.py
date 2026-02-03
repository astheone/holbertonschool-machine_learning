#!/usr/bin/env python3
"""Make predictions using a neural network"""

def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network

    Args:
        network: trained model
        data: input data
        verbose: print progress if True

    Returns:
        prediction results
    """
    return network.predict(data, verbose=verbose)
