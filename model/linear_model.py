import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class layer_4(nn.Module):
    def __init__(self, feature_num, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_stock, embedding_dim=embedding_dim)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_num+embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )

    def forward(self, stock_id, x):
        emb = self.embedding(Tensor(stock_id))
        out_put = self.linear_relu_stack(torch.cat([x, emb], dim=1))
        return F.sigmoid(out_put)
