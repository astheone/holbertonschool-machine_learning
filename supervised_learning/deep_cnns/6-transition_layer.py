#!/usr/bin/env python3
"""Transition Layer for DenseNet-C architecture."""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Build a transition layer as described in DenseNet paper.

    Args:
        X: output from the previous layer
        nb_filters: integer, number of filters in X
        compression: compression factor for the transition layer

    Returns:
        Output of the transition layer and number of filters
    """
    init = K.initializers.HeNormal(seed=0)
    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        nb_filters,
        (1, 1),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(X)

    return X, nb_filters
