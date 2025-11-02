#!/usr/bin/env python3
"""
Sets the Timestamp column as the index of the dataframe
"""


def index(df):
    """Sets Timestamp as index"""
    df = df.set_index('Timestamp')
    return df
