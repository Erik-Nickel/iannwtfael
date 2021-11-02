import numpy as np
from mlp import MLP
from layer import Layer


class LogicalGateMLP(MLP):

    def __init__(self):
        super().__init__(Layer(2, 4), Layer(4, 1))

    def train(self, data, epochs):
        averages = []
        for epoch in range(epochs):
            print('epoch: ', epoch)
            for d in data:
                # print("data: ", d)
                results = [self.__training_step(d[0], d[1])]
                # print('results: ', results)
                av = np.average(results, 0)
                averages.append(av)
                # print('averages: ', averages)
        return averages

    def __training_step(self, input_vector, expected_result):
        (accuracy, loss) = self.backprop_step(input_vector, expected_result)
        print("(accuracy, loss) : ", (accuracy, loss))
        result = np.array([np.round(accuracy[0]), loss[0]])
        print('rounded result: ', result)
        return result

    def classify(self, x, y):
        return np.round(self.forward_step(np.array([x, y]))[0])
