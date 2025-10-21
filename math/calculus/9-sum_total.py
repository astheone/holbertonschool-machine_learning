#!/usr/bin/env python3
def summation_i_squared(n):
    """Return the sum of squares from 1 to n"""
    if type(n) is not int or n < 1:
        return None
    # Formula pa loops: (n * (n + 1) * (2n + 1)) / 6
    return int((n * (n + 1) * (2 * n + 1)) / 6)
