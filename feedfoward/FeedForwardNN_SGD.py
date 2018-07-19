import numpy as np
import random


from feedfoward import subsidiary


class FeedFowrardNN_SGD(object):

    def __init__(self, sizes = [2,3,2], output=True):
        '''

        :param sizes: list of number
        :param output:
        '''

        np.random.seed(1)

        self.num_layers = len(sizes) # count of layer
        self.sizes = sizes # list of layer
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] # list of the biases of matrix
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])] # list of the weight of matrix

        self.output = output
        self.activations = [] # list of activation vector

    def feedforward(self, a):
        '''
        feedforward
        :param a:
        :return:
        '''
        print(a)
        for b, w in zip(self.biases, self.weights):
            a = subsidiary.sigmoid(np.dot(w,a) + b)
            self.activations.append(a) # без учёта входных сиганлов
        return a

    def SGD(self, trainig_data, epochs, mini_batch_size, lr, test_data=None):
        '''
        Stohastic gradient descent
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

    def update_mini_batch(self, mini_batch, lr):
        """
        Обновить веса и смещения нейронной сети, сделав шаг градиентного
        спуска на основе алгоритма обратного распространения ошибки, примененного
        к одному mini batch.
        ``mini_batch`` - список кортежей вида ``(x, y)``,
        ``eta`` - величина шага (learning rate).
        """
        '''
        РАЗОБРАТЬ!
        '''

        '''Создаём список матриц для дельт(коовицентов изменения весов синапсов)'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        for x, y in mini_batch: # проходим по мини пакету
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) # получаем список матриц дельт(коофицентов изменения для матриц весов синапсов)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # прибавляем матрицу дельт к матрице нулевой (состоящей из нулей)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # прибавляем матрицу дельт к матрице нулевой (состоящей из нулей)

        eps = lr / len(mini_batch)
        self.weights = [w - eps * nw for w, nw in zip(self.weights, nabla_w)] # производим обновление матрицы весов
        self.biases = [b - eps * nb for b, nb in zip(self.biases, nabla_b)] # производим обновление матрицы сдвигов

    def backprop(self, x, y):
        '''
        Backpropagation method
        :param x:
        :param y:
        :return:
        '''

        '''список матриц весов и сдвигов'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]# список актвиций нейронов sigma(z^l) = a^l
        zs = []#список перемножений актвиции a^l-1 * w + b

        '''forward propagation'''
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = subsidiary.sigmoid(z)
            activations.append(activation)

        '''получаем дельту (BP1)(частаная производная dC/da)'''
        delta = self.cost_derivative(activations[-1], y) * subsidiary.sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = subsidiary.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)


    def cost_derivative(self, output_activation, y):
        return (output_activation - y)


    def evaludate(self, test_data):
        '''
        Метод позволяющий оценить качество работы нейросети
        :param test_data:
        :return:
        '''
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)


if __name__ == '__main__':
    nn = FeedFowrard([2,3,2])
    x = np.array([1,1], ndmin=2).T
    y = np.array([0,1], ndmin=2).T
    print('x:',x)
    print('y:',y)


    print(nn.feedforward(x))

    nn.SGD(y, 3000, 2, 0.1)