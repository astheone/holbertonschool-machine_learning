#!/usr/bin/env python3
"""
Computes descriptive statistics for all columns except Timestamp
"""

import pandas as pd


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp
    """
    # Hiq kolonën 'Timestamp' nëse ekziston
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    # Llogarit statistikat për kolonat e mbetura
    stats = df.describe()

    # Kthe dataframe-n me statistikat
    return stats
