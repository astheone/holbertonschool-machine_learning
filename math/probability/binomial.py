#!/usr/bin/env python3
""" A scipt that represents a Binomial distribution """


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        """
        A class that represents a Binomial distribution
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
        """Calculates the PMF for a given number of successes"""
        # Sigurohu që k është integer
        if not isinstance(k, int):
            k = int(k)

        # Nëse k është jashtë intervalit [0, n] -> probabiliteti = 0
        if k < 0 or k > self.n:
            return 0

        # Kombinimi (n choose k)
        from math import comb
        coeff = comb(self.n, k)

        # Formula e PMF: C(n,k) * p^k * (1-p)^(n-k)
        p = self.p
        n = self.n
        pmf_value = coeff * (p ** k) * ((1 - p) ** (n - k))

        return pmf_value
