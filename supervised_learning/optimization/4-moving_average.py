#!/usr/bin/env python3
"""Moving Average"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set
    with bias correction"""
    moving_averages = []
    v = 0

    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        bias_correction = 1 - beta ** (i + 1)
        moving_averages.append(v / bias_correction)

    return moving_averages
