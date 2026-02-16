#!/usr/bin/env python3
"""Module to calculate precision for each class in a confusion matrix"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                  represent the correct labels and column indices represent
                  the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the precision of each
        class
    """
    # Sum along axis 0 to get total predicted instances per class (column sums)
    true_positives = np.diagonal(confusion)
    total_predicted_per_class = np.sum(confusion, axis=0)

    # Calculate precision (PPV = TP / (TP + FP))
    # For each class: diagonal element / sum of that column
    return true_positives / total_predicted_per_class
