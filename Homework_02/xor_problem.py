import numpy as np
import matplotlib.pyplot as plt
from logical_gate_mlp import LogicalGateMLP


# TODO: data generation
def generator():
    for i in [1, 0]:
        for y in [0, 1]:
            yield np.array([i, y]), np.array([i ^ y])


inputs = [[0, 0], [0, 1], [1, 1], [1, 0]]
labels = [i[0] ^ i[1] for i in inputs]
data = list(zip(inputs, labels))

if __name__ == '__main__':
    mlp = LogicalGateMLP()
    for x in [1, 0]:
        for y in [0, 1]:
            print(f"{x} xor {y} = {mlp.classify(x, y)}")
    accuracy_loss = mlp.train(data, 1000)
    for x in [1, 0]:
        for y in [0, 1]:
            print(f"{x} xor {y} = {mlp.classify(x, y)}")

    # TODO: prettify (labels dont work)
    plt.plot(range(1000), np.array(accuracy_loss).T[0], label='accuracy')
    plt.plot(range(1000), np.array(accuracy_loss).T[1], label='loss')
    # x axe = epochs
    plt.show()
