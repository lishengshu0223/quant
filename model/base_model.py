from typing import Union

import pandas as pd
import numpy as np


def train_loop(X_train, y_train):
    """
    每次训练循环
    :param X_train:
    :param y_train:
    :return: 当前训练结果，例：loss: xxxx, 以及当前模型参数权重
    """
    loss = None
    model = None
    msg = f"loss: {loss}"
    return msg, loss, model


def valid_loop(X_valid, y_valid):
    """
    所有验证集结果展示
    :param eval_set:
    :return:  验证集的评价指标，例：auc: xxx
    """
    auc = 0.60  # 举个例子，就是获取一个评判标准的数值
    msg = "auc=xxx, loss=xxx, accuracy=xxx"
    return msg, auc

def predict(X_test):
    return

class BasicModel:
    def __init__(self, batch_size=1024, max_epoch=100, network=None):
        # 专门针对模型的初始化数值, 例如一些超参数
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.netwok = network  # 需要用到的神经网络

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.DataFrame, np.ndarray],
        eval_set: list[tuple],
    ):
        """
        同sklearn，用于训练出模型参数
        :param X_train:
        :param y_train:
        :param eval_set: 验证集，用于提早结束迭代
        :return:
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values

        self.X_train = X_train
        self.y_train = y_train

        # 第一组元素提早作为早结束的评判标准，其余组元素仅展示评价指标
        self.X_valid = eval_set[0][0]
        self.y_valid = eval_set[0][0]

        max_auc = [0, 0, self.netwok]
        for i in range(self.max_epoch):
            train_msg, loss, network = train_loop(self.X_train, self.y_train)
            valid_msg, auc = valid_loop(self.X_valid, self.y_valid)
            print(f"epoch {i} {train_msg} {valid_msg}")
            # 如果评判标准创新高，则获取新高的循环次数和新高值
            # 如果5轮内未创新高，则提早结束循环
            if auc > max_auc[1]:
                max_auc = [i, auc, network]
            elif i - max_auc[0] >= 5:
                break
        self.netwok = max_auc[2]


    def predict(self, X_test):
        """

        :param X_test: 测试集
        :return:
        """
        self.X_test = X_test

    def predict_proba(self, X_test):
        """

        :param X_test: 测试集
        :return: 模型不同类的概率
        """
        self.X_test = X_test