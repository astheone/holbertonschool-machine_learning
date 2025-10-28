#!/usr/bin/env python3
"""Normal distribution module (Holberton style: no numpy, fixed e & pi)."""


class Normal:
    """Represents a normal (Gaussian) distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution.

        Args:
            data (list): sample data to estimate mean and stddev (population).
            mean (float): mean if data is None.
            stddev (float): standard deviation if data is None.

        Raises:
            TypeError: if data is provided but not a list.
            ValueError: if data has fewer than 2 values.
            ValueError: if stddev <= 0 when data is None.
        """
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
            mu = sum(data) / n
            # population variance (divide by n)
            var = sum((x - mu) ** 2 for x in data) / n

            self.mean = float(mu)
            self.stddev = float(var ** 0.5)

        # constants (Holberton project uses these fixed values)
        self.e = 2.7182818285
        self.pi = 3.1415926536

    def z_score(self, x):
        """Return the z-score of x."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Return the x-value corresponding to z."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """PDF of Normal at x."""
        a = 1 / (self.stddev * (2 * self.pi) ** 0.5)
        expo = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return a * (self.e ** expo)

    def cdf(self, x):
        """CDF of Normal at x using erf approximation (series)."""
        # t = (x - mu) / (sigma * sqrt(2))
        t = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # erf(t) ≈ (2/√π) * (t - t^3/3 + t^5/10 - t^7/42 + t^9/216)
        t2 = t * t
        t3 = t * t2
        t5 = t3 * t2
        t7 = t5 * t2
        t9 = t7 * t2

        sqrt_pi = self.pi ** 0.5
        erf = (2 / sqrt_pi) * (t - t3 / 3 + t5 / 10 - t7 / 42 + t9 / 216)

        return 0.5 * (1 + erf)
git status