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
    # kontrollo që poly të jetë listë me numra (int ose float)
    if (not isinstance(poly, list) or
            not all(isinstance(x, (int, float)) for x in poly)):
        return None

    # nëse është bosh ose me 1 element (si konstante)
    if len(poly) <= 1:
        return [0]

    # llogarit derivatën pa përdorur loops
    derivative = [i * poly[i] for i in range(1, len(poly))]

    # nëse të gjithë elementët janë 0
    if all(v == 0 for v in derivative):
        return [0]

    return derivative
