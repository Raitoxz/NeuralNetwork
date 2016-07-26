# coding: utf-8
import numpy as np


def sigmoid(z):
    """
    sigmoid函数
    :param z: 计算的net值
    :return:函数值
    """
    try:
        return 1 / (1 + np.exp(-z))
    except RuntimeWarning:
        print z
        exit()


def sigmoid_prime(z):
    """
    对sigmoid函数求导
    :param z: 计算的net值
    :return: 导数值
    """
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


def tanh(z):
    """
    tanh函数
    :param z: 计算的net值
    :return:函数值
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
