#!/usr/bin/env python3
"""
Removes rows with NaN values in the 'Close' column
"""


def prune(df):
    """
    Removes entries where 'Close' has NaN values.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        pd.DataFrame: The modified DataFrame
    """
    pruned_df = df.dropna(subset=['Close'])
    return pruned_df
