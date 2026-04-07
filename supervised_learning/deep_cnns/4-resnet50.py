#!/usr/bin/env python3
"""ResNet-50"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture"""
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', kernel_initializer=init
    )(X)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(act1)

    x = projection_block(pool1, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    x = projection_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    x = projection_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    x = projection_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    avg = K.layers.AveragePooling2D((7, 7), strides
