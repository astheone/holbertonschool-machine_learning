#!/usr/bin/env python3
"""DenseNet-121 architecture implementation."""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Build the DenseNet-121 architecture.

    Args:
        growth_rate: the growth rate
        compression: the compression factor

    Returns:
        the keras model
    """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    bn = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(
        64,
        (7, 7),
        strides=2,
        padding='same',
        kernel_initializer=init
    )(act)
    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(conv)

    nb_filters = 64

    pool, nb_filters = dense_block(pool, nb_filters, growth_rate, 6)
    pool, nb_filters = transition_layer(pool, nb_filters, compression)

    pool, nb_filters = dense_block(pool, nb_filters, growth_rate, 12)
    pool, nb_filters = transition_layer(pool, nb_filters, compression)

    pool, nb_filters = dense_block(pool, nb_filters, growth_rate, 24)
    pool, nb_filters = transition_layer(pool, nb_filters, compression)

    pool, nb_filters = dense_block(pool, nb_filters, growth_rate, 16)

    avg = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=1
    )(pool)

    output = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=init
    )(avg)

    return K.models.Model(inputs=X, outputs=output)
