import numpy as np


class Relu:
    """
    Rectified linear unit activation function class
    """

    def __init__(self):
        self.output = None
        self.inputs = None
        self.dinputs = None

    def forward(self, inputs):
        """
        Computes the output of the ReLU, which is just max(0, input) for all inputs
        :param inputs: the input list
        :return: the same list after applying max(0, input) to each value in it
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        # set 0 gradiant when input is negative
        self.dinputs[self.inputs <= 0] = 0

        return self.dinputs


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

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            # since every input affects every output for derivative of softmax function due to the normalization in the equation,
            # need partial derivative of all combinations of both input vectors for backward pass
            # more here: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
