#!/usr/bin/env python3
"""
Fills missing values and cleans up the dataframe
"""


def fill(df):
    """Cleans and fills the DataFrame"""
    # 1. Hiq kolonën Weighted_Price
    df = df.drop(columns=['Weighted_Price'])

    # 2. Plotëso kolonën Close me vlerën e mëparshme (forward fill)
    df['Close'] = df['Close'].fillna(method='ffill')

    # 3. Plotëso kolonat High, Low, Open me vlerën e Close të po asaj rreshti
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4. Plotëso boshët në Volume_(BTC) dhe Volume_(Currency) me 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
