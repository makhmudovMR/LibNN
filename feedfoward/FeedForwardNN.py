import numpy as np


class FeedForwardNN(object):

    def __init__(self, X, y, counteHiddenLayer, countHiddenNodes, output_nodes = 1):

        '''init weight'''
        self.input_layer = np.random.rand(countHiddenNodes, len(X))
        self.hidden_layer = np.array([np.random.rand(countHiddenNodes, countHiddenNodes) for i in range(counteHiddenLayer)])
        self.output_layer = np.random.rand(output_nodes, countHiddenNodes) # wrong

        self.X = np.array(X)
        self.y = np.array(y, ndmin=2).T


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self):
        forward_result = self.forward()


    def forward(self):
        hidden_layer_input = np.dot(self.input_layer, self.X)
        flag = True
        for indx, wl in enumerate(self.hidden_layer):
            if flag:
                input_for_layer = hidden_layer_input
            input_for_layer = np.dot(wl, input_for_layer)
            output_layer = self._sigmoid(input_for_layer)
        return self._sigmoid(np.dot(self.output_layer, output_layer))


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

    f = FeedForwardNN(X, y, 1, 2)
    print(f.forward())

if __name__ == '__main__':
    main()