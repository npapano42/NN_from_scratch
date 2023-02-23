
class SGD:
    """
    Stochastic Gradient Descent optimizer
    """
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """
        Updates the parameters of a layer based on the learning rate
        :param layer: a layer of the neural network
        """
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
