#!/usr/bin/env python3
"""
Loads data from a file into a pandas DataFrame
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file
    Args:
        filename (str): The name of the file to load from
        delimiter (str): The column separator
    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
