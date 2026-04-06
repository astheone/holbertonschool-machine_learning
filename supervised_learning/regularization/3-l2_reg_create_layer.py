#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a neural network layer with L2 regularization"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=regularizer
    )
    return layer(prev)
