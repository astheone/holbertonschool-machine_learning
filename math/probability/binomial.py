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
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
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

        # Define factorial manually (no imports allowed)
        def factorial(x):
            """Calculates factorial of x."""
            if x == 0 or x == 1:
                return 1
            fact = 1
            for i in range(2, x + 1):
                fact *= i
            return fact

        # Combination formula: C(n, k) = n! / (k! * (n - k)!)
        coeff = factorial(self.n) / (factorial(k) * factorial(self.n - k))

        # PMF formula: C(n, k) * p^k * (1 - p)^(n - k)
        p = self.p
        n = self.n
        pmf_value = coeff * (p ** k) * ((1 - p) ** (n - k))

        return pmf_value
