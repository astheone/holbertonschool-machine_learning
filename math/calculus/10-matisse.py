#!/usr/bin/env python3
"""
This module contains a function that calculates
the derivative of a polynomial represented as a list.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    poly: list of coefficients (index = power of x)
    Returns a new list with the derivative coefficients.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None
    if len(poly) == 1:
        return [0]

    derivative = [i * poly[i] for i in range(1, len(poly))]
    if not derivative or all(v == 0 for v in derivative):
        return [0]
    return derivative
