import numpy as np


class FeedForwardNN(object):

    def __init__(self, X, y, counteHiddenLayer, countHiddenNodes):
        self.input_layer = np.random.rand(countHiddenNodes, len(X))
        self.hidden_layer = np.array([np.random.rand(countHiddenNodes, countHiddenNodes) for i in range(counteHiddenLayer)])
        self.output_layer = np.random.rand(y.shape[0], countHiddenNodes)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self):
        pass

    def forward(self):
        pass


def main():
    X = np.array([
        [0,0],
        [1,0],
        [0,1],
        [1,1]
    ])
    y = np.array([0, 0, 0, 1], ndmin=2).T

    # print(X)
    # print(y)

    f = FeedForwardNN(X, y, 3, 3)
    print(f.hidden_layer)


if __name__ == '__main__':
    main()