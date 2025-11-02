#!/usr/bin/env python3
"""
Loads data from a file into a pandas DataFrame.
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
    # Lexo vetëm 1 milion rreshta për të shmangur memory crash
    try:
        df_iter = pd.read_csv(filename, delimiter=delimiter, chunksize=100000)
        df = next(df_iter)  # Merr vetëm chunk-un e parë
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    return df
