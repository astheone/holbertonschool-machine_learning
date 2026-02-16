#!/usr/bin/env python3
"""Module to calculate sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    
    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                  represent the correct labels and column indices represent
                  the predicted labels
    
    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity of each
        class
    """
    # Sum along axis 1 to get total true instances per class (row sums)
    true_positives = np.diagonal(confusion)
    total_per_class = np.sum(confusion, axis=1)
    
    # Calculate sensitivity (TPR = TP / (TP + FN))
    # For each class: diagonal element / sum of that row
    return true_positives / total_per_class
