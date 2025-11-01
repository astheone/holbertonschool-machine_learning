#!/usr/bin/env python3
"""
Creates a pandas DataFrame from a NumPy ndarray
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.
    Columns are labeled alphabetically in uppercase.
    """
    cols = [chr(65 + i) for i in range(array.shape[1])]
    df = pd.DataFrame(array, columns=cols)
    return df
