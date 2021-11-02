import numpy as np
from logical_gate_mlp import LogicalGateMLP


def generator():
    for i in [1, 0]:
        for y in [0, 1]:
            yield np.array([i, y]), np.array([i ^ y])


if __name__ == '__main__':
    mlp = LogicalGateMLP()
    print(mlp.forward_step(np.array([1, 0])))
    mlp.train(generator(), 10)
