#!/usr/bin/env python3
"""Matrix shape utility."""


def matrix_shape(matrix):
    """Return the shape of a nested list (matrix) as a list of integers."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
