#!/usr/bin/env python3
"""
Sorts and transposes a DataFrame
"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order
    and transposes the sorted DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        pd.DataFrame: The transformed DataFrame
    """
    sorted_df = df.sort_index(ascending=False)
    transposed_df = sorted_df.transpose()
    return transposed_df
