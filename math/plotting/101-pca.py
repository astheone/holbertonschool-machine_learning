#!/usr/bin/env python3
"""
Module 101-pca
This module contains a function that performs PCA
and displays a 3D scatter plot of the Iris dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pca():
    """
    Loads the Iris dataset from pca.npz,
    performs PCA and plots the data in 3D space.
    """
    lib = np.load("pca.npz")
    data = lib["data"]
    labels = lib["labels"]

    # Normalize data (mean-centering)
    mean = np.mean(data, axis=0)
    norm_data = data - mean

    # Perform SVD (for PCA)
    _, _, Vh = np.linalg.svd(norm_data)
    W = Vh[:3].T
    pca_data = np.matmul(norm_data, W)

    # 3D Visualization
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        pca_data[:, 2],
        c=labels,
        cmap="plasma",
        s=50
    )

    ax.set_title("PCA of Iris Dataset")
    ax.set_xlabel("U1")
    ax.set_ylabel("U2")
    ax.set_zlabel("U3")
    plt.show()
