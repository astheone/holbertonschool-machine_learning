#!/usr/bin/env python3
import numpy as np
from from_numpy import from_numpy

np.random.seed(0)
A = np.random.randn(5, 8)
print(from_numpy(A))

B = np.random.randn(9, 3)
print(from_numpy(B))
