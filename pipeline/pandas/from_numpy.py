#!/usr/bin/env python3
"""
Creates a pandas DataFrame from a NumPy ndarray
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray
    The columns of the DataFrame should be labeled in alphabetical order and capitalized.
    """
    import string
    columns = list(string.ascii_uppercase[:array.shape[1]])  # A, B, C, D...
    df = pd.DataFrame(array, columns=columns)
    return df
