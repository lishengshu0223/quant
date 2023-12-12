import random
from typing import Callable, Tuple, Any, Union

import torch
import torch.nn as nn
import numpy as np

from .activations import entmax15, entmoid15, sparsemax, sparsemoid


class GateFullyConnect(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cat_dims: list[int],
        cat_idxs: list[int],
        cat_emb_dim: Union[int, list[int]],
        n_stages: int,
        shrink: int = 4,
        dropout: float = 0.0,
        output_dim: int = 2,
        feature_mask_function: Callable = entmax15,
    ):
        super(GateFullyConnect, self).__init__()
        self.embedding = CategoryEmbedding(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        post_embed_dim = self.embedding.post_embed_dim
        self.gflu = GatedFeatureLearningUnit(
            post_embed_dim, n_stages, feature_mask_function, dropout
        )
        self.fc = FullyConnect(post_embed_dim, output_dim, dropout, shrink)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.gflu(x)
        x = self.fc(x)
        return x


class GatedAdditiveTreesBackbone(nn.Module):
    ACTIVATION_MAP = {
        "entmax": entmax15,
        "sparsemax": sparsemax,
        "softmax": nn.functional.softmax,
    }

    BINARY_ACTIVATION_MAP = {
        "entmoid": entmoid15,
        "sparsemoid": sparsemoid,
        "sigmoid": nn.functional.sigmoid,
    }

    def __init__(
        self,
        cat_embedding_dims: list,
        n_continuous_features: int,
        gflu_stages: int,
        num_trees: int,
        tree_depth: int,
        chain_trees: bool = True,
        tree_wise_attention: bool = False,
        tree_wise_attention_dropout: float = 0.0,
        gflu_dropout: float = 0.0,
        tree_dropout: float = 0.0,
        binning_activation: str = "entmoid",
        feature_mask_function: str = "softmax",
        batch_norm_continuous_input: bool = True,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        assert (
            binning_activation in self.BINARY_ACTIVATION_MAP.keys()
        ), f"`binning_activation should be one of {self.BINARY_ACTIVATION_MAP.keys()}"
        assert (
            feature_mask_function in self.ACTIVATION_MAP.keys()
        ), f"`feature_mask_function should be one of {self.ACTIVATION_MAP.keys()}"

        self.gflu_stages = gflu_stages
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.chain_trees = chain_trees
        self.tree_wise_attention = tree_wise_attention
        self.tree_wise_attention_dropout = tree_wise_attention_dropout
        self.gflu_dropout = gflu_dropout
        self.tree_dropout = tree_dropout
        self.binning_activation = self.BINARY_ACTIVATION_MAP[binning_activation]
        self.feature_mask_function = self.ACTIVATION_MAP[feature_mask_function]
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.n_continuous_features = n_continuous_features
        self.cat_embedding_dims = cat_embedding_dims
        self._embedded_cat_features = sum([y for x, y in cat_embedding_dims])
        self.n_features = self._embedded_cat_features + n_continuous_features
        self.embedding_dropout = embedding_dropout
        self.output_dim = 2 ** self.tree_depth
        self._build_network()

    def _build_network(self):
        if self.gflu_stages > 0:
            self.gflus = GatedFeatureLearningUnit(
                n_features_in=self.n_features,
                n_stages=self.gflu_stages,
                feature_mask_function=self.feature_mask_function,
                dropout=self.gflu_dropout,
            )
        self.trees = nn.ModuleList(
            [
                NeuralDecisionTree(
                    depth=self.tree_depth,
                    n_features=self.n_features + 2 ** self.tree_depth * t
                    if self.chain_trees
                    else self.n_features,
                    dropout=self.tree_dropout,
                    binning_activation=self.binning_activation,
                    feature_mask_function=self.feature_mask_function,
                )
                for t in range(self.num_trees)
            ]
        )
        if self.tree_wise_attention:
            self.tree_attention = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=1,
                batch_first=False,
                dropout=self.tree_wise_attention_dropout,
            )
        self.eta = nn.Parameter(torch.rand(self.num_trees, requires_grad=True))
        self.fc = nn.Linear(2 ** self.tree_depth, 2)

    def _build_embedding_layer(self):
        return Embedding1dLayer(
            continuous_dim=self.n_continuous_features,
            categorical_embedding_dims=self.cat_embedding_dims,
            embedding_dropout=self.embedding_dropout,
            batch_norm_continuous_input=self.batch_norm_continuous_input,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gflu_stages > 0:
            x = self.gflus(x)
        # Decision Tree
        tree_outputs = []
        tree_feature_masks = []  # TODO make this optional and create feat importance
        tree_input = x
        for i in range(self.num_trees):
            tree_output, feat_masks = self.trees[i](tree_input)
            tree_outputs.append(tree_output.unsqueeze(-1))
            tree_feature_masks.append(feat_masks)
            if self.chain_trees:
                tree_input = torch.cat([tree_input, tree_output], 1)
        tree_outputs = torch.cat(tree_outputs, dim=-1)
        if self.tree_wise_attention:
            tree_outputs = tree_outputs.permute(2, 0, 1)
            tree_outputs, _ = self.tree_attention(
                tree_outputs, tree_outputs, tree_outputs
            )
            tree_outputs = tree_outputs.permute(1, 2, 0)
        tree_outputs = self.fc(tree_outputs.transpose(2, 1))
        tree_outputs = tree_outputs * self.eta.reshape(1, -1, 1)
        tree_outputs = tree_outputs.sum(dim=1)
        return tree_outputs

    @property
    def feature_importance_(self):
        return (
            self.gflus.feature_mask_function(self.gflus.feature_masks)
            .sum(dim=0)
            .detach()
            .cpu()
            .numpy()
        )


class NeuralDecisionStump(nn.Module):
    def __init__(
        self,
        n_features: int,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
    ):
        super().__init__()
        self._num_cutpoints = 1
        self._num_leaf = 2
        self.n_features = n_features
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self._build_network()

    def _build_network(self):
        if self.feature_mask_function is not None:
            # sampling a random beta distribution
            # random distribution helps with diversity in trees and feature splits
            alpha = random.uniform(0.5, 10.0)
            beta = random.uniform(0.5, 10.0)
            # with torch.no_grad():
            feature_mask = (
                torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
                .sample((self.n_features,))
                .squeeze(-1)
            )
            self.feature_mask = nn.Parameter(feature_mask, requires_grad=True)
        W = torch.linspace(
            1.0,
            self._num_cutpoints + 1.0,
            self._num_cutpoints + 1,
            requires_grad=False,
        ).reshape(1, 1, -1)
        self.register_buffer("W", W)

        cutpoints = torch.rand([self.n_features, self._num_cutpoints])
        # Append zeros to the beginning of each row
        cutpoints = torch.cat(
            [torch.zeros([self.n_features, 1], device=cutpoints.device), cutpoints], 1
        )
        self.cut_points = nn.Parameter(cutpoints, requires_grad=True)
        self.leaf_responses = nn.Parameter(
            torch.rand(self.n_features, self._num_leaf), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_mask = self.feature_mask_function(self.feature_mask, 0)
        # Repeat W for each batch size using broadcasting
        W = torch.ones(x.size(0), 1, 1, device=x.device) * self.W
        # Binning features
        x = torch.bmm(x.unsqueeze(-1), W) - self.cut_points.unsqueeze(0)
        x = self.binning_activation(x)  # , dim=-1)
        x = x * self.leaf_responses.unsqueeze(0)
        x = (x * feature_mask.reshape(1, -1, 1)).sum(dim=1)
        return x, feature_mask


class NeuralDecisionTree(nn.Module):
    def __init__(
        self,
        depth: int,
        n_features: int,
        dropout: float = 0,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
    ):
        super().__init__()
        self.depth = depth
        self._num_cutpoints = 1
        self.n_features = n_features
        self._dropout = dropout
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self._build_network()

    def _build_network(self):
        for d in range(self.depth):
            for n in range(max(2 ** (d), 1)):
                self.add_module(
                    f"decision_stump_{d}_{n}",
                    NeuralDecisionStump(
                        self.n_features + (2 ** (d) if d > 0 else 0),
                        self.binning_activation,
                        self.feature_mask_function,
                    ),
                )
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[list]]:
        tree_input = x
        feature_masks = []
        layer_nodes = []
        for d in range(self.depth):
            layer_nodes = []
            layer_feature_masks = []
            for n in range(max(2 ** (d), 1)):
                leaf_nodes, feature_mask = self._modules[f"decision_stump_{d}_{n}"](
                    tree_input
                )
                layer_nodes.append(leaf_nodes)
                layer_feature_masks.append(feature_mask)
            layer_nodes = torch.cat(layer_nodes, dim=1)
            tree_input = torch.cat([x, layer_nodes], dim=1)
            feature_masks.append(layer_feature_masks)
        out = self.dropout(layer_nodes)
        return out, feature_masks


class GatedFeatureLearningUnit(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_stages: int,
        feature_mask_function: Callable = entmax15,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_in
        self.feature_mask_function = feature_mask_function
        self._dropout = dropout
        self.n_stages = n_stages
        self._build_network()

    def _create_feature_mask(self):
        feature_masks = torch.cat(
            [
                torch.distributions.Beta(
                    torch.tensor([random.uniform(0.5, 10.0)]),
                    torch.tensor([random.uniform(0.5, 10.0)]),
                )
                .sample((self.n_features_in,))
                .squeeze(-1)
                for _ in range(self.n_stages)
            ]
        ).reshape(self.n_stages, self.n_features_in)
        return nn.Parameter(feature_masks, requires_grad=True,)

    def _build_network(self):
        self.W_in = nn.ModuleList(
            [
                nn.Linear(2 * self.n_features_in, 2 * self.n_features_in)
                for _ in range(self.n_stages)
            ]
        )
        self.W_out = nn.ModuleList(
            [
                nn.Linear(2 * self.n_features_in, self.n_features_in)
                for _ in range(self.n_stages)
            ]
        )

        self.feature_masks = self._create_feature_mask()
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for d in range(self.n_stages):
            feature = self.feature_mask_function(self.feature_masks[d], 0) * x
            h_in = self.W_in[d](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(h_in[:, : self.n_features_in])
            r = torch.sigmoid(h_in[:, self.n_features_in :])  # noqa: E203
            h_out = torch.tanh(self.W_out[d](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1 - z) * h + z * h_out)
        return h


class Embedding1dLayer(nn.Module):
    """Enables different values in a categorical features to have different embeddings."""

    def __init__(
        self,
        continuous_dim: int,
        categorical_embedding_dims: Tuple[int, int],
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.batch_norm_continuous_input = batch_norm_continuous_input

        # Embedding layers
        self.cat_embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in categorical_embedding_dims]
        )
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        # Continuous Layers
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(continuous_dim)

    def forward(self, x: dict[str, Any]) -> torch.Tensor:
        assert (
            "continuous" in x or "categorical" in x
        ), "x must contain either continuous and categorical features"
        # (B, N)
        continuous_data, categorical_data = (
            x.get("continuous", torch.empty(0, 0)),
            x.get("categorical", torch.empty(0, 0)),
        )
        assert categorical_data.shape[1] == len(
            self.cat_embedding_layers
        ), "categorical_data must have same number of columns as categorical embedding layers"
        assert (
            continuous_data.shape[1] == self.continuous_dim
        ), "continuous_data must have same number of columns as continuous dim"
        embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(continuous_data)
            else:
                embed = continuous_data
            # (B, N, C)
        if categorical_data.shape[1] > 0:
            categorical_embed = torch.cat(
                [
                    embedding_layer(categorical_data[:, i])
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ],
                dim=1,
            )
            # (B, N, C + C)
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed


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

