import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from my_model.linear_model import layer_4 as Mymodel
from my_dataset.linear_dataset import MyDataset


def model_train(interval=5):
    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_dataloader)
        for _, _, x, y in pbar:
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"epoch:{epoch}")
        if epoch % interval == 0:
            model_test()
            torch.save(model.state_dict(), f"log/model_{epoch}.pth")
            torch.save(optimizer.state_dict(), f"log/optimizer_{epoch}.pth")


def model_test():
    with torch.no_grad():
        test_loss = 0
        accuracy_list = []
        pbar = tqdm(test_dataloader)
        for _, _, x, y in pbar:
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            y_label = (y_pred > 0.5).int()
            y = y.int()
            accuracy = accuracy_score(y_label.cpu().numpy(), y.cpu().numpy())
            accuracy_list.append(accuracy)
            test_loss += loss.item()
            pbar.set_description("test")
        test_loss /= len(test_dataloader)
        accuracy = np.mean(accuracy_list)
        print(f"test_lost:{test_loss:.4f}")
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


epochs = 50
batch_size = 1024
learning_rate = 1e-3

print("loading data")

factor_concat_train = pd.read_pickle("F:\\Neural_Networks\data\\factor_concat_2017_2018.pkl")
factor_concat_test = pd.read_pickle("F:\\Neural_Networks\data\\factor_concat_2018_2019_test.pkl")
stock_return = pd.read_pickle("./data/stock_return.pkl")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_num = len(factor_concat_test.columns.levels[0])

model = Mymodel(feature_num)
model.to(device)
print("data setting")
train_dataset = MyDataset(factor_concat_train, stock_return)
print("train dataset")
test_dataset = MyDataset(factor_concat_test, stock_return)
print("test dataset")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
model_train(3)
