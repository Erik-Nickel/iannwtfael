import numpy as np
from functools import reduce


class MLP:

    def __init__(self, *layers):
        self.__layers = layers

    def forward_step(self, input_vector):
        return np.array(reduce(lambda i, l: l.forward_step(i), self.__layers, input_vector))

    # TODO: implement
    def backprop_step(self, error):
        return None

    # TODO: Temporary (to orient on)
    def adapt(self, x, t):
        outputs = [x]
        accum_value = x
        for layer in self.__layers:
            accum_value = layer.activate(accum_value)
            outputs.append(accum_value)

        e = t - outputs[-1]
        for l, i in np.flip(list(enumerate(self.__layers, 1))):
            d = (outputs[i] * (1 - outputs[i])) * e
            e = (l.weight_matrix @ d)[1:]
            l.adapt(outputs[i - 1], d)
