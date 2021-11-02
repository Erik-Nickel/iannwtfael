import numpy as np
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
    print("first try: ", mlp.forward_step(np.array([1, 0])))
    print("first try: ", mlp.forward_step(np.array([1, 1])))
    mlp.train(data, 1000)
    print("after training", mlp.forward_step(np.array([1, 0])))
    print("after training", mlp.forward_step(np.array([1, 1])))
