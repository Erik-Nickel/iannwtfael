import numpy as np


class Perceptron:

    def __init__(self, input_units, alpha, activation_function):
        self.__weights = np.random.rand(input_units + 1)
        self.__alpha = alpha
        self.__activation_function = activation_function

    def forward_step(self, input_vector):
        return self.__activation_function(np.dot(np.insert(input_vector, 0, 1), self.__weights))

    def update(self, input_vector, delta):
        self.__weights = self.__weights + self.__alpha * delta * np.insert(input_vector, 0, 1)

    def weights(self):
        return self.__weights
