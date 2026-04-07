#!/usr/bin/env python3
"""Transfer learning with CIFAR-10 using EfficientNetB0."""
import numpy as np
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Pre-process CIFAR-10 data for EfficientNetB0.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) with CIFAR 10 data
        Y: numpy.ndarray of shape (m,) with CIFAR 10 labels

    Returns:
        X_p: preprocessed X
        Y_p: preprocessed Y
    """
    X_p = K.applications.efficientnet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_val, Y_val) = (
        K.datasets.cifar10.load_data()
    )

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_val, Y_val = preprocess_data(X_val, Y_val)

    inputs = K.Input(shape=(32, 32, 3))
    scaled = K.layers.Lambda(
        lambda x: K.backend.resize_images(
            x,
            height_factor=7,
            width_factor=7,
            data_format='channels_last',
            interpolation='bilinear'
        )
    )(inputs)

    base_model = K.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=scaled,
        pooling='avg'
    )
    base_model.trainable = False

    x = base_model.output
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            verbose=1
        ),
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        X_train, Y_train,
        batch_size=128,
        epochs=20,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, Y_train,
        batch_size=128,
        epochs=10,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )

    model.save('cifar10.h5')
