import numpy as np


class Dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeroes((1, neurons))

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

