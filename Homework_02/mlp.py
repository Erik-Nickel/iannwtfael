import numpy as np
from functools import reduce


def squared_error(expected, output):
    return np.square(expected - output)


class MLP:

    def __init__(self, *layers, loss_fun=squared_error):
        self.__layers = layers
        self.__loss_fun = loss_fun

    def forward_step(self, input_vector):
        return np.array(reduce(lambda i, l: l.forward_step(i), self.__layers, input_vector))

    def backprop_step(self, input_vector, expected):
        outputs = reduce(lambda i, l: [*i, l.forward_step(i[-1])], self.__layers, [input_vector])
        e = expected - outputs[-1]  # for layer N
        for layer, i in np.flip(list(enumerate(self.__layers, 1))):
            d = layer.backprop_function(outputs[i]) * e
            e = (layer.weight_matrix()[1:] @ d)  # for next layer ([1:] erases the bias weights)
            layer.update(outputs[i - 1], d)
        return (1 - np.abs(expected - np.round(outputs[-1]))), self.__loss_fun(expected, outputs[-1])
