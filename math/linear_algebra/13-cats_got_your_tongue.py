#!/usr/bin/env python3
"""
This module defines a function that concatenates two numpy.ndarrays
along a specific axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return the concatenation of two numpy.ndarrays along a given axis"""
    return np.concatenate((mat1, mat2), axis)
