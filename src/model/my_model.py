import os
from typing import Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve

from src.dataset.dataset import SectionalDataset


class MyModel:
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

    def _trainloop(self, train_dataloader):
        mean_loss = []
        for x, y in train_dataloader:
            pred = self.network(x)
            loss = self.loss_fn(pred, y.to(torch.long))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            mean_loss.append(loss.item() / len(x))
        mean_loss = np.sum(mean_loss) / len(mean_loss)
        return mean_loss

    def _validationloop(self, test_dataloader):
        predict = []
        true_y = []
        with torch.no_grad():
            for X, y in test_dataloader:
                pred = self.network(X)
                true_y = true_y + y.int().tolist()
                predict = predict + pred[:,-1].tolist()
        fpr, tpr, thresholds = roc_curve(true_y, predict)
        auc_value = auc(fpr, tpr)
        return auc_value

    def save(self, dir, name=None):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if name is not None:
            dir = os.path.join(dir, name)
        else:
            dir = os.path.join(dir, "model")
        torch.save(self.network, f"{dir}.pth")

    def load(self, dir):
        self.network = torch.load(dir)

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
        # 第一组元素提早作为早结束的评判标准，其余组元素仅展示评价指标(目前紧展示第一个元素)
        if eval_set is not None:
            X_valid = eval_set[0][0]
            y_valid = eval_set[0][1]
            valid_dataset = SectionalDataset(X_valid, y_valid)
            valid_dataloader = DataLoader(valid_dataset, self.batch_size)
            del X_valid, y_valid, valid_dataset
            has_valid = True
        else:
            valid_dataloader = None
            has_valid = False

        train_dataset = SectionalDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, self.batch_size)
        del X_train, y_train, train_dataset

        best_auc = 0
        best_epoch = 0
        count = 0
        for i in range(1, self.max_epoch + 1):
            mean_loss = self._trainloop(train_dataloader)
            # 若果有验证集，则计算auc，保存最佳auc和最佳network，若5轮无提升，则循环终止， 返回auc
            if has_valid:
                auc_value = self._validationloop(valid_dataloader)
                print((f"epoch {i}, loss: {mean_loss:.4f}| eval_auc: {auc_value:.4f}"))
                if auc_value > best_auc:
                    self.save("./model", "best_model")
                    best_auc = auc_value
                    best_epoch = i
                    count = 0
                else:
                    count = count + 1
                    if count >= 5:
                        self.load("./model/best_model.pth")
                        print(
                            f"looping has early stopped, best epoch is {best_epoch}, which has auc {best_auc:.4f}"
                        )
                        break
            else:
                print((f"epoch {i}, loss: {mean_loss:.4f}"))

    def predict(self, X_test):
        """

        :param X_test: 测试集
        :return:
        """


    def predict_proba(self, X_test):
        """

        :param X_test: 测试集
        :return: 模型不同类的概率
        """
        X_test = torch.tensor(X_test).to(torch.float)
        y_pred = self.network(X_test)
        y_pred = y_pred.detach().numpy()
        return y_pred