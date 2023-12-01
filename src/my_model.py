import copy
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def calculate_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 的执行时间为: {execution_time} 秒")
        return result

    return wrapper


class MyModel:
    def __init__(
        self,
        batch_size: int,
        max_epoch: int,
        early_stop_epoch: int,
        optimizer,
        loss_fn,
        network,
        data_stype="sectional",
        **model_params
    ):
        network_params = model_params['network_params']
        try:
            optimizer_params = model_params['optimizer_params']
        except:
            optimizer_params = {}
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # 专门针对模型的初始化数值, 例如一些超参数
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.loss_fn = loss_fn()
        self.early_stop_epoch = early_stop_epoch
        self.network = network(**network_params).to(self.device)  # 需要用到的神经网络
        self.optimizer = optimizer(self.network.parameters(), **optimizer_params)
        self.data_stype = data_stype
        self.optimizer = optimizer(self.network.parameters())

    def _trainloop(self, train_dataloader):
        mean_loss = 0
        for x, y in train_dataloader:
            pred = self.network(x.to(self.device))
            loss = self.loss_fn(pred, y.to(torch.long).to(self.device))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            mean_loss += loss.item()
        mean_loss = mean_loss / len(train_dataloader)
        return mean_loss

    def _validationloop(self, test_dataloader):
        mean_loss = 0
        self.network.eval()
        with torch.no_grad():
            for x, y in test_dataloader:
                pred = self.network(x.to(self.device))
                loss = self.loss_fn(pred, y.to(torch.long).to(self.device))
                mean_loss += loss.item()
        mean_loss = mean_loss / len(test_dataloader)
        self.network.train()
        return mean_loss

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
        self, train_dataset: Dataset, eval_dataset: Dataset = None,
    ):
        """

        :param train_dataset:
        :param eval_dataset:
        :return:
        """
        # 第一组元素提早作为早结束的评判标准，其余组元素仅展示评价指标(目前紧展示第一个元素)
        if eval_dataset is not None:
            valid_dataloader = DataLoader(eval_dataset, self.batch_size)
            has_eval = True
        else:
            valid_dataloader = None
            has_eval = False
        train_dataloader = DataLoader(train_dataset, self.batch_size)

        min_loss = 1e5
        best_epoch = 0
        best_network = copy.deepcopy(self.network.to('cpu'))
        count = 0
        for i in range(1, self.max_epoch + 1):
            period_time = time.time()
            train_loss = self._trainloop(train_dataloader)
            period_time = int(time.time() - period_time)
            # 若果有验证集，则计算auc，保存最佳auc和最佳network，若5轮无提升，则循环终止， 返回auc
            if has_eval:
                eval_loss = self._validationloop(valid_dataloader)
                period_time = int(time.time() - period_time)
                minutes = str(period_time // 60).zfill(2)
                seconds = str(period_time % 60).zfill(2)
                msg = f"epoch {i}, train_loss: {train_loss:.4f}| eval_loss: {eval_loss:.4f}| time: {minutes}:{seconds}"
                judge_loss = eval_loss
                # if auc_value > best_auc:
                #     self.save("./model", "best_model")
                #     best_auc = auc_value
                #     best_epoch = i
                #     count = 0
                # else:
                #     count = count + 1
                #     if count >= self.early_stop_epoch:
                #         self.load("./model/best_model.pth")
                #         print(
                #             f"looping has early stopped, best epoch is {best_epoch}, which has auc {best_auc:.4f}"
                #         )
                #         break
            else:
                minutes = str(period_time // 60).zfill(2)
                seconds = str(period_time % 60).zfill(2)
                msg = f"epoch {i}, train_loss: {train_loss:.4f}| time: {minutes}:{seconds}"
                judge_loss = train_loss
            print(msg)
            if judge_loss < min_loss:
                min_loss = judge_loss
                best_network = copy.copy(self.network.to('cpu'))
                best_epoch = i
            else:
                count = count + 1
                if count >= self.early_stop_epoch:
                    msg = f"looping has early stopped, best epoch is {best_epoch}, which has minum loss {min_loss:.4f}"
                    print(msg)
                    break
        self.network = copy.deepcopy(best_network).to(self.device)

    def predict(self, test_dataset: Dataset):
        """

        :param X_test: 测试集
        :return:
        """
        y_pred = self.predict_proba(test_dataset)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def predict_proba(self, test_dataset: Dataset):
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)
        self.network.eval()
        test_pred = []
        with torch.no_grad():
            for x in test_dataloader:
                pred = self.network(x.to(self.device)).to('cpu')
                test_pred.append(pred)
        test_pred = np.vstack(test_pred)
        return test_pred
