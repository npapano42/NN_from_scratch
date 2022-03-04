import numpy as np


class Relu:

    def forward(self, inputs):
        return np.maximum(0, inputs)


class Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

