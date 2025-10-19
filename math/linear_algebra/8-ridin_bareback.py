#!/usr/bin/env python3
"""Performs matrix multiplication between two 2D matrices."""


def mat_mul(mat1, mat2):
    """Multiply two 2D matrices and return the result matrix."""
    # Kontrollo nëse mat1 dhe mat2 mund të shumëzohen
    if len(mat1[0]) != len(mat2):
        return None

    # Krijo një matrix bosh me përmasa [len(mat1)] x [len(mat2[0])]
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            s = 0
            for k in range(len(mat2)):
                s += mat1[i][k] * mat2[k][j]
            row.append(s)
        result.append(row)
    return result
