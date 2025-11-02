#!/usr/bin/env python3
"""
Transforms and visualizes cryptocurrency data
"""

import matplotlib.pyplot as plt
import pandas as pd


def visualize(df):
    """
    Transforms the given DataFrame and plots daily cryptocurrency data
    """
    # 1. Hiq kolonën 'Weighted_Price'
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    # 2. Riemëro 'Timestamp' në 'Date'
    df = df.rename(columns={'Timestamp': 'Date'})

    # 3. Kthe 'Date' në datetime
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    # 4. Vendos 'Date' si index
    df = df.set_index('Date')

    # 5. Plotëso vlerat që mungojnë
    df['Close'] = df['Close'].fillna(method='ffill')
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    # 6. Filtrimi: nga 2017 e lart
    df = df[df.index.year >= 2017]

    # 7. Grupimi ditor me funksione të ndryshme sipas kërkesës
    df = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    # 8. Vizualizo (pa printime)
    df.plot(figsize=(12, 6), title="Cryptocurrency daily data (from 2017)")
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.show()

    # 9. Kthe DataFrame-n e transformuar
    return df
