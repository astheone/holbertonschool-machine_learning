#!/usr/bin/env python3
"""
Creates a hierarchical DataFrame with Timestamp as the first index level
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenate the bitstamp and coinbase tables from timestamps
    1417411980 to 1417417980 inclusive, labeled respectively.
    """
    # Vendos Timestamp si index për të dy dataframe-t
    df1 = index(df1)
    df2 = index(df2)

    # Merr vetëm rreshtat që janë brenda intervalit të kërkuar
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Bashko dataframe-t dhe shto labels
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Kthe rendin e MultiIndex-it që Timestamp të jetë i pari
    df = df.swaplevel().sort_index()

    # Kthe dataframe-n
    return df
