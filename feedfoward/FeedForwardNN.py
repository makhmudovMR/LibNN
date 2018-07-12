import numpy as np
import random


from feedfoward import subsidiary


class FeedFowrard(object):


    def __init__(self, sizes = [2,3,2], output=True):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[:1]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

        self.output = output

        self.activations = []


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = subsidiary.sigmoid(np.dot(w,a) + b)
            self.activations.append(a) # без учёта входных сиганлов
        return a

    def SGD(self, trainig_data, epochs, mini_batch_size, lr, test_data=None):
        pass





if __name__ == '__main__':
    nn = FeedFowrard()