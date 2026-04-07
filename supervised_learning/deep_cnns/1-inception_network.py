#!/usr/bin/env python3
"""Inception Network"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network"""
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', activation='relu'
    )(X)
    pool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(conv1)

    conv2 = K.layers.Conv2D(
        64, (1, 1), padding='same', activation='relu'
    )(pool1)
    conv3 = K.layers.Conv2D(
        192, (3, 3), padding='same', activation='relu'
    )(conv2)
    pool2 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(conv3)

    inc3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inc3b = inception_block(inc3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
