#!/usr/bin/env python3
"""
Slices specific columns and rows from a DataFrame
"""


def slice(df):
    """
    Extracts the columns 'High', 'Low', 'Close', and 'Volume_(BTC)'
    and selects every 60th row.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        pd.DataFrame: The sliced DataFrame
    """
    sliced_df = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
    return sliced_df
