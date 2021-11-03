import numpy as np
from mlp import MLP
from layer import Layer


class LogicalGateMLP(MLP):

    def __init__(self):
        super().__init__(Layer(2, 4), Layer(4, 1))

    def train(self, data, epochs):
        averages = [np.average([self.__training_step(d[0], d[1]) for d in data], 0) for _ in range(epochs)]
        print("avg:", averages)
        return averages

    def __training_step(self, input_vector, expected_result):
        (accuracy, loss) = self.backprop_step(input_vector, expected_result)
        result = np.array([accuracy[0], loss[0]])
        # print('[accuracy, loss]: ', result)
        return result

    def classify(self, x, y):
        return np.round(self.forward_step(np.array([x, y]))[0])
