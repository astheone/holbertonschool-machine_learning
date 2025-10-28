#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Represents a Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the binomial distribution.
        If data is provided, estimates n and p from it.
        Otherwise, uses the given n and p values.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p_first = 1 - (variance / mean)
            n_round = mean / p_first

            self.n = round(n_round)
            self.p = mean / self.n

    def pmf(self, k):
        """
        Calculates the Probability Mass Function (PMF)
        for a given number of successes k.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(x):
            """Simple factorial without math module"""
            if x == 0 or x == 1:
                return 1
            f = 1
            for i in range(2, x + 1):
                f *= i
            return f

        comb = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        if k > self.n:
            k = self.n

        cdf_value = 0
        for i in range(0, k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
