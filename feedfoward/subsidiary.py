import numpy as np


def sigmoid(z):
    '''Sigomid'''
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    '''Derivative of sigmoid function'''
    return sigmoid(z) * (1 - sigmoid(z))


def cost_function(network, test_data, onehot=True):
    c = 0
    for exmaple, y in test_data:
        if not onehot:
            y = np.eye(3,1, k=-int(y))
            yhat = network.feedforward(example)
            c += np.sum((y - yhat) ** 2)

    return c / len(test_data)