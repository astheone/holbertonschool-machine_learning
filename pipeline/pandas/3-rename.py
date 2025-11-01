#!/usr/bin/env python3
"""
Renames a DataFrame column and converts timestamps to datetime
"""

import pandas as pd


def rename(df):
    """
    Renames the 'Timestamp' column to 'Datetime'
    Converts timestamps to datetime and keeps only 'Datetime' and 'Close'
    Args:
        df (pd.DataFrame): The input DataFrame
    Returns:
        pd.DataFrame: The modified DataFrame
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
