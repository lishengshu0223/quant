from typing import Union

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.dataset.dataset import SectionalDataset
from src.model.base_function import train_loop, validation_loop


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
    def __init__(
        self,
        batch_size=1024,
        max_epoch=100,
        optimizer=None,
        loss_fn=None,
        network=None,
        **network_kwargs,
    ):
        # 专门针对模型的初始化数值, 例如一些超参数
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.network = network(**network_kwargs)  # 需要用到的神经网络

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.DataFrame, np.ndarray],
        eval_set: list[tuple] = None,
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

        # 第一组元素提早作为早结束的评判标准，其余组元素仅展示评价指标(目前紧展示第一个元素)
        if eval_set is not None:
            self.X_valid = eval_set[0][0]
            self.y_valid = eval_set[0][0]
            valid_dataset = SectionalDataset(self.X_valid, self.y_valid)
            valid_dataloader = DataLoader(valid_dataset, self.batch_size)
            del valid_dataset
            has_valid = True
        else:
            valid_dataloader = None
            has_valid = False

        train_dataset = SectionalDataset(self.X_train, self.y_train)
        train_dataloader = DataLoader(train_dataset, self.batch_size)
        del train_dataset

        max_auc = 0
        best_network = self.network
        count = 0
        for i in range(1, self.max_epoch+1):
            self.network, self.loss_fn, self.optimizer, mean_loss = train_loop(
                self.network, train_dataloader, self.loss_fn, self.optimizer
            )
            # 若果有验证集，则计算auc，保存最佳auc和最佳network，若5轮无提升，则循环终止， 返回auc
            if has_valid:
                auc = validation_loop(
                    self.network, valid_dataloader, self.loss_fn, self.optimizer
                )
                print((f"epoch {i}, loss: {mean_loss:.4f}| eval_auc: {auc:.4f}"))
                if auc > max_auc:
                    best_network = self.network
                    max_auc = auc
                    count = 0
                else:
                    count = count + 1
                    if count>=5:
                        self.network = best_network
                        break
            else:
                print((f"epoch {i}, loss: {mean_loss:.4f}"))

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
