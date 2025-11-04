#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def line():

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
     
    # your code here
    plt.plot(y, color ='red')
    plt.xlim(0, 10)
    plt.show()
    