#!/usr/bin/env python3
"""
Concatenates two dataframes on their Timestamp index
"""

import pandas as pd
index = __import__('10-index').index

def concat(df1, df2):
    """
    Concatenate df2 and df1 using Timestamp as index
    """
    # 1. Vendos Timestamp si index për të dy dataframe-t
    df1 = index(df1)
    df2 = index(df2)

    # 2. Merr nga df2 (bitstamp) vetëm rreshtat deri në 1417411920
    df2 = df2.loc[:1417411920]

    # 3. Bashko dataframe-t (df2 sipër, df1 poshtë)
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # 4. Kthe dataframe-n e bashkuar
    return df
