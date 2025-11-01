#!/usr/bin/env python3
"""
Converts selected columns from a DataFrame into a NumPy array
"""

import pandas as pd
import numpy as np


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns
    and converts them into a numpy.ndarray
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        np.ndarray: The numpy array containing the selected data
    """
    arr = df[['High', 'Close']].tail(10).to_numpy()
    return arr
