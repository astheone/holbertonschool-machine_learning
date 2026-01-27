#!/usr/bin/env python3
import numpy as np
from 26-deep_neural_network import DeepNeuralNetwork

# Dummy data (përdor të njëjtin si test script)
np.random.seed(0)
X = np.random.randn(5, 10)  # 5 features, 10 examples
Y = np.random.randint(0, 2, (1, 10))

# Krijo rrjetin neural
nn = DeepNeuralNetwork(nx=5, layers=[3, 1])

# Forward propagation
A1, cache = nn.forward_prop(X)

# Print output as test pret
np.set_printoptions(precision=8)
print(nn.L)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
print(A1)
