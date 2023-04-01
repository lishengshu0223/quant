import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader

# from my_model.linear_model import layer_4 as Mymodel
from my_model.linear_model import layer_embedding as Mymodel
from my_dataset.linear_dataset import MyDataset, MyFastDataset


def model_train(interval=5):
    for epoch in range(1, epochs + 1):
        print(epoch)
        # pbar = tqdm(train_dataloader)
        accuracy_list = []
        train_loss = 0
        for _, code_num, x, y in train_dataloader:
            optimizer.zero_grad()
            code_num = code_num.int().to(device)
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = model(code_num, x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            y_label = (y_pred > 0.5).int()
            y = y.int()
            accuracy = accuracy_score(y_label.cpu().numpy(), y.cpu().numpy())
            accuracy_list.append(accuracy)
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        accuracy = np.mean(accuracy_list)
        print(f"train_loss:{train_loss:.4f}")
        print(f"accuracy:{accuracy:.4f}")

        if epoch % interval == 0:
            model_test()
            torch.save(model.state_dict(), f"log/18_eb/model_{epoch}.pth")
            torch.save(optimizer.state_dict(), f"log/18_eb/optimizer_{epoch}.pth")


def model_test():
    with torch.no_grad():
        test_loss = 0
        accuracy_list = []
        # pbar = tqdm(test_dataloader)
        for _, code_num, x, y in test_dataloader:
            code_num = code_num.int().to(device)
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = model(code_num, x)
            loss = criterion(y_pred, y)
            y_label = (y_pred > 0.5).int()
            y = y.int()
            accuracy = accuracy_score(y_label.cpu().numpy(), y.cpu().numpy())
            accuracy_list.append(accuracy)
            test_loss += loss.item()
            # pbar.set_description("test")
        test_loss /= len(test_dataloader)
        accuracy = np.mean(accuracy_list)
        print(f"test_loss:{test_loss:.4f}")
        print(f"accuracy:{accuracy:.4f}")


# model, optimizer = train(
#     factor_concat,
#     stock_return,
#     pd.Timestamp(2017, 1, 1),
#     pd.Timestamp(2018, 1, 1),
# )
# modelstate = torch.load("./log/log.pth")
# checkpoint = torch.load("./log/optimizer.pth")

# torch.save(model.state_dict(), "log/model_2018.pth")
# torch.save(optimizer.state_dict(), "log/optimizer_2018.pth")


epochs = 30
batch_size = 256
learning_rate = 1e-3

print("loading data")

factor_stack_train = pd.read_pickle("./data/processed_data/factor_stack_2015_2018.pkl")
factor_stack_test = pd.read_pickle(
    "./data/processed_data/factor_stack_2018_2019_test.pkl"
)
quantile_return = pd.read_pickle("./data/processed_data/quantile_return.pkl")
date_num_dict = pd.read_pickle("./data/processed_data/date_num_dict.pkl")
code_num_dict = pd.read_pickle("./data/processed_data/date_num_dict.pkl")
num_date_dict = pd.read_pickle("./data/processed_data/num_date_dict.pkl")
num_code_dict = pd.read_pickle("./data/processed_data/num_code_dict.pkl")

print("loading model")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_num = len(factor_stack_train.columns)
stock_num = len(num_code_dict)
embedding_dim = 10
model = Mymodel(feature_num, stock_num, embedding_dim)
model.to(device)

print("train dataset")
train_dataset = MyFastDataset(
    factor_stack=factor_stack_train,
    quantile_return=quantile_return,
    date_num_dict=date_num_dict,
    code_num_dict=code_num_dict,
    num_date_dict=num_date_dict,
    num_code_dict=num_code_dict,
)
del factor_stack_train
print("test dataset")
test_dataset = MyFastDataset(
    factor_stack=factor_stack_test,
    quantile_return=factor_stack_test,
    date_num_dict=date_num_dict,
    code_num_dict=code_num_dict,
    num_date_dict=num_date_dict,
    num_code_dict=num_code_dict,
)
del factor_stack_test

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
model_train(3)
