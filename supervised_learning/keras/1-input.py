#!/usr/bin/env python3
"""
Module for building a neural network with Keras using Functional API
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using Functional API

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        the keras model
    """
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.L2(lambtha)

    x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=regularizer
    )(inputs)

    if len(layers) > 1:
        x = K.layers.Dropout(1 - keep_prob)(x)

    for i in range(1, len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer
        )(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
