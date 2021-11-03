import numpy as np
import matplotlib.pyplot as plt
from logical_gate_mlp import LogicalGateMLP


# TODO: data generation random
def generator():
    for x in [1, 0]:
        for y in [0, 1]:
            yield [x, y], x ^ y


if __name__ == '__main__':
    mlp = LogicalGateMLP()
    for x in [1, 0]:
        for y in [0, 1]:
            print(f"{x} xor {y} = {mlp.classify(x, y)}")
    accuracy_loss = mlp.train(list(generator()), 1000)
    for x in [1, 0]:
        for y in [0, 1]:
            print(f"{x} xor {y} = {mlp.classify(x, y)}")

    plt.plot(range(1000), np.array(accuracy_loss).T[0], label="Training accuracy")
    plt.plot(range(1000), np.array(accuracy_loss).T[1], label="Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    plt.show()
