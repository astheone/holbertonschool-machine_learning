#!/usr/bin/env python3
"""
Computes descriptive statistics for all columns except Timestamp
"""


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp
    """
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    return df.describe()
