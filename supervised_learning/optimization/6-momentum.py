#!/usr/bin/env python3
"""Momentum Upgraded"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Sets up gradient descent with momentum optimization in TensorFlow"""
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
