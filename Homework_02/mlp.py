import numpy as np
from functools import reduce


# TODO: square error is not signed :(
def squared_error(expected, output):
    return expected - output
    # return np.square(expected - output)


class MLP:

    def __init__(self, *layers, error_fun=squared_error):
        self.__layers = layers
        self.__error_fun = error_fun

    def forward_step(self, input_vector):
        return np.array(reduce(lambda i, l: l.forward_step(i), self.__layers, input_vector))

    def backprop_step(self, input_vector, expected):
        outputs = reduce(lambda i, l: [*i, l.forward_step(i[-1])], self.__layers, [input_vector])
        # TODO: binary outputs?
        error = self.__error_fun(expected, outputs[-1])
        e = error  # for layer N
        for layer, i in np.flip(list(enumerate(self.__layers, 1))):
            d = layer.activation_function_prime(outputs[i]) * e
            e = (layer.weight_matrix()[1:] @ d)  # for next layer ([1:] erases the bias weights)
            layer.update(outputs[i - 1], d)
        return (1 - np.abs(expected - outputs[-1])), np.abs(error)
