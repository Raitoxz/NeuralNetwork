# coding: utf-8
import numpy as np
from ActivationFunction import sigmoid
import Reader


class NeuralNetwork:
    def __init__(self):
        self.input_layer_size = 15  # size of input layer
        self.hidden_layer_size = 20  # size of hidden layer
        self.output_layer_size = 10  # size of output layer

        # weights between input layer and hidden layer
        self.W1 = np.random.randn(self.hidden_layer_size, self.input_layer_size)
        # weights between hidden layer and output layer
        self.W2 = np.random.randn(self.output_layer_size, self.hidden_layer_size)

        self.n = 10  # number of sample

        self.x = np.zeros((self.n, self.input_layer_size), dtype='float64')  # value of input in sample
        self.y = np.zeros((self.n, self.output_layer_size), dtype='float64')  # value of output in sample

        self.i = np.zeros((self.input_layer_size, 1), dtype='float64')  # nodes of input layer
        self.h = np.zeros((self.n, self.hidden_layer_size), dtype='float64')  # nodes of hidden layer
        self.o = np.zeros((self.output_layer_size, 1), dtype='float64')  # nodes of output layer

        self.learning_rate = 0.5  # 学习率
        self.loss = 0  # 损失
        self.lambda_ = 0.0001

    def read_data(self, file_name):
        """
        将读取的数据保存在array中
        :return:
        """
        data = Reader.read_data(file_name)
        for i in range(self.n):
            self.x[i] = np.array(data[i][:15])
            self.y[i] = np.array(data[i][15:])

    def forward_propagation(self, turn):
        """
        前馈计算
        :param turn: 计算第turn个样本数据
        :return:
        """
        self.i = self.x[turn]
        self.h = sigmoid(np.dot(self.W1, self.i))
        self.o = sigmoid(np.dot(self.W2, self.h))

    def back_propagation(self, turn):
        """
        反馈计算
        :param turn: 计算第turn个样本数据
        :return:
        """
        for i in range(self.output_layer_size):
            for j in range(self.hidden_layer_size):
                # 这里有一个问题，self.o是0-1值，还是离散的值
                # 计算 dJdW2ij
                try:
                    self.W2[i][j] += self.learning_rate * self.o[i] * (1 - self.o[i]) * (self.y[turn][i] - self.o[i]) \
                                     * self.h[j] + self.W2[i][j] * self.lambda_
                except IndexError:
                    print i, j
                    exit()

        for i in range(self.hidden_layer_size):
            for j in range(self.input_layer_size):
                temp = 0
                for k in range(self.output_layer_size):
                    temp += (self.y[turn][k] - self.o[k]) * self.o[k] * (1 - self.o[k]) * self.W2[k][i]
                # 计算dJdW1ij
                self.W1[i][j] += self.learning_rate * self.i[j] * self.h[i] * (1 - self.h[i]) * temp + self.W1[i][j] \
                                * self.lambda_

    def loss_function(self, turn):
        """
        损失函数
        :return:
        """
        self.loss = 0
        for i in range(self.output_layer_size):
            self.loss += (self.o[i] - self.y[turn][i]) ** 2
        self.loss *= 0.5

    def train(self):
        """
        训练模型
        :return:
        """
        for i in range(self.n):
            self.forward_propagation(i)
            self.back_propagation(i)

    def test(self, file_name):
        """
        测试
        :return:
        """
        self.read_data(file_name)
        count = 0
        for i in range(self.n):
            h = sigmoid(np.dot(self.W1, self.x[i]))
            o = sigmoid(np.dot(self.W2, h))
            predict = list(o).index(max(list(o)))
            if predict == i:
                count += 1
        print count * 1.0 / 10


if __name__ == '__main__':
    a = NeuralNetwork()
    a.read_data('data.txt')
    for i_ in range(10000):
        a.train()
        print '第', i_, '次迭代，正确率是:',
        a.test('test.txt')
