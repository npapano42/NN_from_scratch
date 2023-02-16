import numpy as np


class Relu:
    """
    Rectified linear unit activation function class
    """

    def forward(self, inputs):
        """
        Computes the output of the ReLU, which is just max(0, input) for all inputs
        :param inputs: the input list
        :return: the same list after applying max(0, input) to each value in it
        """
        return np.maximum(0, inputs)


class Softmax:
    """
    Softmax activation function class
    """

    def forward(self, inputs):
        """
        Computes the output of the softmax function, which exponentiates the output then returns the normalized values
        :param inputs: the input list
        :return: the list after applying the softmax function
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

