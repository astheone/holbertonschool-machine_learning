#!/usr/bin/env python3
"""
This module defines a function that performs element-wise
addition, subtraction, multiplication, and division of two
numpy.ndarrays.
"""


def np_elementwise(mat1, mat2):
    """Return element-wise sum, difference, product and quotient
    of two numpy.ndarrays"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
