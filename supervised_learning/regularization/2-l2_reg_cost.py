#!/usr/bin/env python3
"""L2 Regularization Cost with Keras"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates cost of a neural network with L2 regularization"""
    l2_costs = []
    for layer in model.layers:
        if layer.losses:
            l2_costs.append(cost + tf.reduce_sum(layer.losses))
    return tf.stack(l2_costs)
