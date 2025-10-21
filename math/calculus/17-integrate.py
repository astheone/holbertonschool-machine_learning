#!/usr/bin/env python3
"""Module that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.
    Args:
        poly (list): coefficients representing the polynomial
        C (int, optional): integration constant
    Returns:
        list: coefficients of the integral of the polynomial
    """
    if (not isinstance(poly, list) or len(poly) == 0 or
            not isinstance(C, (int, float))):
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    result = [C]
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        if val.is_integer():
            val = int(val)
        result.append(val)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
