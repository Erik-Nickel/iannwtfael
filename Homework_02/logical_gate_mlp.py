import numpy as np
from mlp import MLP
from layer import Layer


def sqr(res, exp):
    return np.square(res - exp)


class LogicalGateMLP(MLP):

    def __init__(self):
        super().__init__(Layer(2, 4), Layer(4, 1))

    def train(self, data, epochs, error_fun=sqr):
        for a in data:
            print(a)
        print("ddd")
        lll = []
        for e in range(epochs):
            print(e)
            rr = [self.__training_step(d[0], d[1]) for d in data]
            print(rr)
            print("aaa")
            av = np.average(rr, 0)
            lll.append(av)
            print(lll)
            print("bbb")
        return lll

    def __training_step(self, input_vector, expected_result, error_fun=sqr):
        res = self.forward_step(input_vector)
        error = error_fun(res, expected_result)
        self.backprop_step(error)
        res = np.array([np.round(res)[0], error[0]])
        print(res)
        return res
