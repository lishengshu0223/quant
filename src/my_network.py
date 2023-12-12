import math
from typing import Union

import numpy as np
import torch
from torch import nn


class CategoryEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cat_dims: list[int],
        cat_idxs: list[int],
        cat_emb_dim: Union[int, list[int]],
    ):
        """
        @param input_dim: int
            输入的特征数
        @param cat_dims: list of int
             以类别作为特征维度（每个类有多少中不同的分类，例：性别分位男女两类）
        @param cat_idxs: list of int
            以类为特征的特征所在输入数据的坐标
        @param cat_emb_dim: int or list of int
            每个类特征的Embedding的维度，如果是int则每个类有相同的Embedding维度
        """
        super(CategoryEmbedding, self).__init__()
        if cat_dims is None:
            cat_dims = []
        if cat_idxs is None:
            cat_idxs = []
        if cat_emb_dim is None:
            cat_emb_dim = []
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
        elif (cat_dims == []) ^ (cat_idxs == []):
            if cat_dims == []:
                msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
            else:
                msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
            raise ValueError(msg)
        elif len(cat_dims) != len(cat_idxs):
            msg = "The lists cat_dims and cat_idxs must have the same length."
            raise ValueError(msg)
        else:
            self.skip_embedding = False

        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        if len(self.cat_emb_dims) != len(cat_dims):
            msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(
            input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims)
        )

        self.embeddings = torch.nn.ModuleList()

        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        if self.skip_embedding:
            return x
        else:
            cols = []
            cat_feat_counter = 0
            for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
                if is_continuous:
                    cols.append(x[:, feat_init_idx].float().view(-1, 1))
                else:
                    cols.append(
                        self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                    )
                    cat_feat_counter += 1
            post_embeddings = torch.cat(cols, dim=1)
            return post_embeddings


class FullyConnect(nn.Module):
    def __init__(self, feature_num: int, output_dim: int, p: float = 0.5, shrink: int = 4):
        super(FullyConnect, self).__init__()
        self.feature_num = feature_num
        self.fc_block = nn.ModuleList()
        while True:
            if int(feature_num / shrink) < output_dim:
                break
            fc_block = nn.Sequential(
                nn.Linear(feature_num, int(feature_num / shrink)),
                nn.ReLU(),
                nn.BatchNorm1d(int(feature_num / shrink)),
                nn.Dropout(p),
            )
            self.fc_block.append(fc_block)
            feature_num = int(feature_num / shrink)
        if feature_num != output_dim:
            fc_block = nn.Sequential(
                nn.Linear(feature_num, output_dim), nn.ReLU(), nn.BatchNorm1d(output_dim), nn.Dropout(p)
            )
            self.fc_block.append(fc_block)

    def forward(self, x):
        for nn in self.fc_block:
            x = nn(x)
        return x


class Embedding_Fc(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cat_dims: list[int] = None,
        cat_idxs: list[int] = None,
        cat_emb_dim: Union[int, list[int]] = None,
        p: float = 0.5,
        shrink: int = 4,
    ):
        super(Embedding_Fc, self).__init__()
        self.embedding = CategoryEmbedding(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.fc = FullyConnect(self.embedding.post_embed_dim, output_dim, p, shrink)

    def forward(self, x):
        post_emdedding = self.embedding(x)
        out = self.fc(post_emdedding)
        return out


class GruNetwork(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=30,
                 num_layers=1,
                 num_labels=2,
                 bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False):
        super(GruNetwork, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.bn(x[:, -1, :])
        out = self.linear(x)
        return out


class GruNetworkWithAttention(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=30,
                 num_layers=1,
                 num_labels=2,
                 bias=True,
                 batch_first=True,
                 dropout=0.0,
                 bidirectional=False):
        super(GruNetworkWithAttention, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=hidden_size, out_features=int(hidden_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(hidden_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))
        self.linear = nn.Linear(2 * hidden_size, num_labels)


    def forward(self, x):
        x, _ = self.gru(x)
        attention_score = self.att_net(x)
        att = torch.mul(x, attention_score)
        att = torch.sum(att, dim=1)
        cat = torch.cat((x[:, -1, :], att), dim=1)
        cat = self.bn(cat)
        out = self.linear(cat)
        return out


class MyTabTransformer(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, p: float = 0.1, shrink: int = 16
    ):
        super(MyTabTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = 2 ** (int(math.log(input_dim, 2)) + 1)
        self.linear = nn.Linear(self.input_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=8, dropout=p
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fullyconnect = FullyConnect(self.d_model, self.output_dim, p, shrink)

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer(x)
        out = self.fullyconnect(x)
        return out

if __name__ == '__main__':
    model = CategoryEmbedding(1147, [2, 31, 12], [0, 1, 2], [4, 6, 8])
