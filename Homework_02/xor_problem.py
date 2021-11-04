import numpy as np
import matplotlib.pyplot as plt
from logical_gate_mlp import LogicalGateMLP


def data_gen(n):
    for _ in range(n):
        for x in [1, 0]:
            for y in [0, 1]:
                yield [x, y], x ^ y


if __name__ == '__main__':
    mlp = LogicalGateMLP()
    accuracy_loss = mlp.train(list(data_gen(16)), 1000)
    plt.plot(range(1000), np.array(accuracy_loss).T[0], label="Training accuracy")
    plt.plot(range(1000), np.array(accuracy_loss).T[1], label="Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    plt.show()
