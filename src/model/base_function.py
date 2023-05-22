import numpy as np
import torch
from sklearn import metrics

def train_loop(network, dataloader, loss_fn, optimizer):
    mean_loss = []
    for x, y in dataloader:
        pred = network(x).flatten()
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_loss.append(loss.item() / len(x))
    mean_loss = np.sum(mean_loss) / len(mean_loss)
    return network, loss_fn, optimizer, mean_loss


def validation_loop(network, dataloader, loss_fn, optimizer):
    predict = []
    true_y = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = network(X)
            true_y = true_y + y.int().tolist()
            predict = predict + pred.tolist()
    fpr, tpr, thresholds = metrics.roc_curve(true_y, predict)
    auc = metrics.auc(fpr, tpr)
    return auc