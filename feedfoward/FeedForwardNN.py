import numpy as np
import random


from feedfoward import subsidiary


class FeedFowrard(object):

    def __init__(self, sizes = [2,3,2], output=True):
        '''

        :param sizes:
        :param output:
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[:1]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

        self.output = output
        self.activations = []

    def feedforward(self, a):
        '''

        :param a:
        :return:
        '''
        for b, w in zip(self.biases, self.weights):
            a = subsidiary.sigmoid(np.dot(w,a) + b)
            self.activations.append(a) # без учёта входных сиганлов
        return a

    def SGD(self, trainig_data, epochs, mini_batch_size, lr, test_data=None):
        '''
        Разобрвть ещё раз!!!
        :param trainig_data:
        :param epochs:
        :param mini_batch_size:
        :param lr:
        :param test_data:
        :return:
        '''

        if test_data is not None:
            n_test = len(test_data)
        n = len(trainig_data)
        success_tests = 0
        for j in range(epochs):
            random.shuffle(trainig_data)
            # формируем мини-пакет
            mini_batches = [trainig_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

            '''мониторинг эволюции нейронной сети'''
            if test_data is not None and self.output:
                success_tests = self.evaluate(test_data)
                print('Эпоха {0}: {1} / {2}'.format(j, success_tests, n_test))
            elif self.output:
                print('Эпоха {0} завершена'.format(j))
        if test_data is not None:
            return success_tests / n_test

if __name__ == '__main__':
    nn = FeedFowrard()