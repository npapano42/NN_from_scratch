import numpy as np


class Loss:
    """
    Master loss class for others to inherit from
    """

    def calculate(self, output, y):
        """
        Calculates the final loss value, which is the average of all the sample losses
        :param output: the output of a layer
        :param y: the expected value
        :return: the loss value
        """
        sample_losses = self.forward(output, y)

        return np.mean(sample_losses)

    def forward(self, output, y):
        pass


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross Entropy Loss
    """

    def __init__(self):
        self.dinputs = None

    def forward(self, y_pred, y_true):
        """
        Forward pass of the CCE loss function. Works for categorical and one-hot encoded labels
        :param y_pred: the predictions
        :param y_true: the actual values
        :return: the negative log likelihood
        """

        # clip data to prevent log(0) causing math problems
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probability for target (for categorical labels, where y_true is a single list of values)
        if len(y_true.shape) == 1:
            # for each pred, get the value of the true class
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]

        # mask values (for one-hot encoded labels, where each value in y_true is also a list)
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        else:
            raise Exception(f"Shape of y_true is incorrect! Shape: {y_true.shape}")

        # return the negative log
        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        """
        Backwards pass of CCE. Computes the normalized gradiant, converting y_true if needed to one-hot encoded vector
        :param dvalues: derivative values
        :param y_true: actual values
        :return: the normalized gradiant
        """
        # if labels are sparse, convert to one-hot vector
        if len(y_true.shape) == 1:
            # eye creates a list of one-hot encoded vectors based on the passed in int
            y_true = np.eye(len(dvalues[0]))[y_true]

        # normalized gradiant
        self.dinputs = (-y_true / dvalues) / len(dvalues)

        return self.dinputs



