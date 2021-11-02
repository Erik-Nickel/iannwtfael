import numpy as np
from functools import reduce


# TODO: square error is not signed :(
def squared_error(output, expected):
    return output - expected
    # return np.square(expected - output)


def sigmoid_prime(x):
    return x * (1 - x)


class MLP:

    def __init__(self, *layers, error_fun=squared_error, activation_fun_prime=sigmoid_prime):
        self.__layers = layers
        self.__error_fun = error_fun
        self.__activation_fun_prime = activation_fun_prime

    def forward_step(self, input_vector):
        return np.array(reduce(lambda i, l: l.forward_step(i), self.__layers, input_vector))

    # TODO: experimental
    def backprop_step(self, input_vector, expected):
        outputs = [input_vector]

        accum_value = input_vector
        for layer in self.__layers:
            accum_value = layer.forward_step(accum_value)
            outputs.append(accum_value)

        loss = self.__error_fun(expected, outputs[-1])

        e = loss  # for layer N
        for layer, i in np.flip(list(enumerate(self.__layers, 1))):
            # print("e: ", e)
            d = self.__activation_fun_prime(outputs[i]) * e
            # print("d:", d)
            # print("wight matrix: ", layer.weight_matrix())
            e = (layer.weight_matrix()[1:] @ d)  # for next layer ([1:] erases the bias weights)
            layer.update(outputs[i - 1], d)

        return (1 - np.abs(expected - outputs[-1])), loss
