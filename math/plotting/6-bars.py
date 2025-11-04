#!/usr/bin/env python3
"""
Module 6-bars
This module contains a function that plots a stacked bar graph
representing the number of fruits different people possess.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph showing the number of fruits
    (apples, bananas, oranges, peaches) each person has.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    plt.figure(figsize=(6.4, 4.8))

    bottom = np.zeros(3)
    for i in range(len(fruit)):
        plt.bar(people, fruit[i], width=0.5,
                color=colors[i], label=fruits[i], bottom=bottom)
        bottom += fruit[i]

    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
