import numpy as np


class Dense:
    """
    Dense layer, where everything is fully connected
    """

    def __init__(self, inputs, neurons):
        """
        Creates a dense layer with the given input and neuron size
        :param inputs: the size of the input
        :param neurons: the number of neurons to include
        """
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeroes((1, neurons))

    def forward(self, inputs):
        """
        Forward pass through the layer, computing the dot product and adding the biases
        :param inputs: the input to the layer as a list
        :return: the output of the layer
        """
        return np.dot(inputs, self.weights) + self.biases

