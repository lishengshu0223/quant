# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
#
# from my_model.linear_model import layer_4 as Mymodel
# from my_dataset.linear_dataset import MyDataset
# from my_dataset.timeseries_dataset import MyFastDataset
#
#
# def model_train(interval=5):
#     for epoch in range(1, epochs + 1):
#         print(epoch)
#         # pbar = tqdm(train_dataloader)
#         accuracy_list = []
#         train_loss = 0
#         for _, _, x, y in train_dataloader:
#             optimizer.zero_grad()
#             x = x.float().to(device)
#             y = y.float().to(device)
#             y_pred = model(x)
#             loss = criterion(y_pred, y)
#             loss.backward()
#             optimizer.step()
#
#             y_label = (y_pred > 0.5).int()
#             y = y.int()
#             accuracy = accuracy_score(y_label.cpu().numpy(), y.cpu().numpy())
#             accuracy_list.append(accuracy)
#             train_loss += loss.item()
#
#         train_loss /= len(train_dataloader)
#         accuracy = np.mean(accuracy_list)
#         print(f"train_loss:{train_loss:.4f}")
#         print(f"accuracy:{accuracy:.4f}")
#
#         if epoch % interval == 0:
#             model_test()
#             torch.save(model.state_dict(), f"log/18_eb/model_{epoch}.pth")
#             torch.save(optimizer.state_dict(), f"log/18_eb/optimizer_{epoch}.pth")
#
#
# def model_test():
#     with torch.no_grad():
#         test_loss = 0
#         accuracy_list = []
#         # pbar = tqdm(test_dataloader)
#         for _, _, x, y in test_dataloader:
#             x = x.float().to(device)
#             y = y.float().to(device)
#             y_pred = model(x)
#             loss = criterion(y_pred, y)
#             y_label = (y_pred > 0.5).int()
#             y = y.int()
#             accuracy = accuracy_score(y_label.cpu().numpy(), y.cpu().numpy())
#             accuracy_list.append(accuracy)
#             test_loss += loss.item()
#             # pbar.set_description("test")
#         test_loss /= len(test_dataloader)
#         accuracy = np.mean(accuracy_list)
#         print(f"test_loss:{test_loss:.4f}")
#         print(f"accuracy:{accuracy:.4f}")
#
#
# # model, optimizer = train(
# #     factor_concat,
# #     stock_return,
# #     pd.Timestamp(2017, 1, 1),
# #     pd.Timestamp(2018, 1, 1),
# # )
# # modelstate = torch.load("./log/log.pth")
# # checkpoint = torch.load("./log/optimizer.pth")
#
# # torch.save(model.state_dict(), "log/model_2018.pth")
# # torch.save(optimizer.state_dict(), "log/optimizer_2018.pth")
#
#
# epochs = 90
# batch_size = 256
# learning_rate = 1e-3
#
# print("loading data")
#
# # factor_stack_train = pd.read_pickle("./data/processed_data/factor_stack_2015_2018.pkl")
# # factor_stack_test = pd.read_pickle(
# #     "./data/factor_concat_2019_2020.pkl"
# # ).stack()
# # quantile_return = pd.read_pickle("./data/processed_data/quantile_return.pkl")
# # date_num_dict = pd.read_pickle("./data/processed_data/date_num_dict.pkl")
# # code_num_dict = pd.read_pickle("./data/processed_data/code_num_dict.pkl")
# # num_date_dict = pd.read_pickle("./data/processed_data/num_date_dict.pkl")
# # num_code_dict = pd.read_pickle("./data/processed_data/num_code_dict.pkl")
# #
# # print("loading model")
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # feature_num = len(factor_stack_train.columns)
# # stock_num = len(num_code_dict)
# # embedding_dim = 400
# # model = Mymodel(feature_num, stock_num, embedding_dim)
# # model.to(device)
# #
# # print("train dataset")
# # train_dataset = MyFastDataset(
# #     factor_stack=factor_stack_train,
# #     quantile_return=quantile_return,
# #     date_num_dict=date_num_dict,
# #     code_num_dict=code_num_dict,
# #     num_date_dict=num_date_dict,
# #     num_code_dict=num_code_dict,
# # )
# # del factor_stack_train
# # print("test dataset")
# # test_dataset = MyFastDataset(
# #     factor_stack=factor_stack_test,
# #     quantile_return=quantile_return,
# #     date_num_dict=date_num_dict,
# #     code_num_dict=code_num_dict,
# #     num_date_dict=num_date_dict,
# #     num_code_dict=num_code_dict,
# # )
# # del factor_stack_test
# # del quantile_return
# # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
# # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # criterion = nn.BCEWithLogitsLoss()
# # model_train(3)
#
# data_train = pd.read_pickle("./data/processed_data/value_dict.pkl")
# data_test = pd.read_pickle("./data/processed_data/value_dict_test.pkl")
# quantile_return = pd.read_pickle("./data/processed_data/quantile_return.pkl")
# date_num_dict = pd.read_pickle("./data/processed_data/date_num_dict.pkl")
# code_num_dict = pd.read_pickle("./data/processed_data/code_num_dict.pkl")
# num_date_dict = pd.read_pickle("./data/processed_data/num_date_dict.pkl")
# num_code_dict = pd.read_pickle("./data/processed_data/num_code_dict.pkl")
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# feature_num = 975 * 5
# model = Mymodel(feature_num)
# model.to(device)
#
# print("train dataset")
# train_dataset = MyFastDataset(
#     value_dict=data_train,
#     quantile_return=quantile_return,
#     date_num_dict=date_num_dict,
#     code_num_dict=code_num_dict,
#     num_date_dict=num_date_dict,
#     num_code_dict=num_code_dict,
# )
# test_dataset = MyFastDataset(
#     value_dict=data_test,
#     quantile_return=quantile_return,
#     date_num_dict=date_num_dict,
#     code_num_dict=code_num_dict,
#     num_date_dict=num_date_dict,
#     num_code_dict=num_code_dict,
# )
# del data_train
# del quantile_return
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.BCEWithLogitsLoss()
# model_train(3)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier


selected_factor_dict = pd.read_pickle("./data/selected_factor_dict.pkl")

max_epochs = 100
for year in [2018, 2019, 2020, 2021, 2022]:
    print(year)
    x_train = pd.read_pickle(f"./data/processed_data/factor_stack_{year - 3}.pkl")
    x_train = pd.concat(
        [x_train, pd.read_pickle(f"./data/processed_data/factor_stack_{year - 2}.pkl")]
    )
    x_train = pd.concat(
        [x_train, pd.read_pickle(f"./data/processed_data/factor_stack_{year - 1}.pkl")]
    )
    y_train = pd.read_pickle(f"./data/processed_data/quantile_return_{year - 3}.pkl")
    y_train = pd.concat(
        [
            y_train,
            pd.read_pickle(f"./data/processed_data/quantile_return_{year - 2}.pkl"),
        ]
    )
    y_train = pd.concat(
        [
            y_train,
            pd.read_pickle(f"./data/processed_data/quantile_return_{year - 1}.pkl"),
        ]
    )

    selected_factor = selected_factor_dict[year]
    x_train = x_train[selected_factor]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train.values, y_train.values
    )
    tabnet_params = dict(
        optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
    )
    model = TabNetClassifier(**tabnet_params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        max_epochs=max_epochs,
        drop_last=False,
    )
    model.save_model(f"./log/tabnet/{year}")
    del (
        x_train,
        x_valid,
        y_train,
        y_valid,
    )

    x_test = pd.read_pickle(f"./data/processed_data/factor_stack_{year}.pkl")
    x_test = x_test[selected_factor]

    score = model.predict_proba(x_test.values)
    score = pd.DataFrame(score[:, 1], index=pd.MultiIndex.from_tuples(x_test.index))
    score.index.names = ["dt", "code"]
    score = score.unstack()
    score.columns = score.columns.droplevel(0)
    score.to_pickle(f"F:\Multifactor_Project\score_{year}.pkl")
