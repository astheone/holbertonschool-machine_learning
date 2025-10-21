#!/usr/bin/env python3
"""
This module contains a function that calculates
the summation of i squared from 1 to n.
"""

def summation_i_squared(n):
    """
    Return the sum of squares from 1 to n.
    If n is not a valid positive integer, return None.
    """
    if type(n) is not int or n < 1:
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)
