#!/usr/bin/env python3
"""Module to calculate specificity for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                  represent the correct labels and column indices represent
                  the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity of each
        class
    """
    # True positives for each class (diagonal elements)
    true_positives = np.diagonal(confusion)

    # Total actual instances per class (sum of each row)
    total_actual_per_class = np.sum(confusion, axis=1)

    # Total predicted instances per class (sum of each column)
    total_predicted_per_class = np.sum(confusion, axis=0)

    # Total number of all instances
    total_instances = np.sum(confusion)

    # For each class, calculate TN, FP, FN
    # TN = total - (TP + FP + FN)
    # FP = sum of column - TP
    # FN = sum of row - TP
    false_positives = total_predicted_per_class - true_positives
    false_negatives = total_actual_per_class - true_positives

    # True negatives for each class
    true_negatives = total_instances - (true_positives + false_positives +
                                        false_negatives)

    # Specificity = TN / (TN + FP)
    return true_negatives / (true_negatives + false_positives)
