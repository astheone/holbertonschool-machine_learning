#!/usr/bin/env python3
"""Batch Normalization Upgraded"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in TensorFlow"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')

    mean, variance = tf.nn.moments(dense, axes=[0])
    z_norm = tf.nn.batch_normalization(
        dense, mean, variance, beta, gamma, variance_epsilon=1e-7
    )

    return activation(z_norm)
