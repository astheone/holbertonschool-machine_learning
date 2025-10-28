#!/usr/bin/env python3
"""Normal distribution module (Holberton project)"""


class Normal:
    """Represents a normal (Gaussian) distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            mean = sum(data) / n
            variance = sum((x - mean) ** 2 for x in data) / n
            self.mean = float(mean)
            self.stddev = float(variance ** 0.5)

        self.e = 2.7182818285
        self.pi = 3.1415926536

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        a = 1 / (self.stddev * ((2 * self.pi) ** 0.5))
        b = (-1 / 2) * ((x - self.mean) / self.stddev) ** 2
        return a * (self.e ** b)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        t = (x - self.mean) / (self.stddev * (2 ** 0.5))
        # erf(t) approximation using Taylor series
        erf = (2 / (self.pi ** 0.5)) * (
            t - (t ** 3) / 3 + (t ** 5) / 10 - (t ** 7) / 42 + (t ** 9) / 216
        )
        return 0.5 * (1 + erf)
