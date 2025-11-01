#!/usr/bin/env python3
"""
Sorts a DataFrame by the 'High' column in descending order
"""


def high(df):
    """
    Sorts the DataFrame by the 'High' column in descending order.
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        pd.DataFrame: The DataFrame sorted by 'High' in descending order
    """
    sorted_df = df.sort_values(by='High', ascending=False)
    return sorted_df
