import numpy as np
from perceptron import Perceptron


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def backprop_sigmoid(x):
    return x * (1 - x)


class Layer:

    def __init__(self, input_units, layer_size, alpha=1, activation_function=sigmoid,
                 backprop_function=backprop_sigmoid):
        self.__perceptrons = [Perceptron(input_units, alpha, activation_function) for _ in range(layer_size)]
        self.__backprop_function = backprop_function

    def forward_step(self, x):
        return np.array([p.forward_step(x) for p in self.__perceptrons])

    def update(self, input_vector, deltas):
        for i, p in enumerate(self.__perceptrons):
            p.update(input_vector, deltas[i])

    def weight_matrix(self):
        return np.asarray([p.weights() for p in self.__perceptrons]).T

    def backprop_function(self, x):
        return self.__backprop_function(x)
