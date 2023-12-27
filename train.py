import os
import torch

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.tuner import Tuner
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import lightning as L
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc
from src.network import GateFullyConnect
from src.my_dataset import SectionalTrainDataset, SectionalPredictionDataset

from pytorch_lightning import seed_everything

# Set seed
seed = 42
seed_everything(seed)
torch.set_float32_matmul_precision("medium")


class LitGate(L.LightningModule):
    def __init__(
        self,
        input_dim,
        cat_dims=None,
        cat_idxs=None,
        cat_emb_dim=None,
        n_stages=20,
        shrink=512,
    ):
        super().__init__()
        self.gate = GateFullyConnect(
            input_dim, cat_dims, cat_idxs, cat_emb_dim, n_stages, shrink
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = 1e-5
        self.lr = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.gate(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.gate(x)
        pred_proba = torch.softmax(y_hat, dim=1)[:, 1]
        auc = roc_auc_score(y.cpu().numpy(), pred_proba.cpu().numpy(), labels=1)
        self.log('val_auc', auc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        y_hat = self.gate(x)
        pred_proba = torch.softmax(y_hat, dim=1)[:, 1]
        return pred_proba

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.learning_rate)
        )
        return optimizer


class LitQTDM(TQDMProgressBar):
    def __init__(self):
        super(LitQTDM, self).__init__()

    def init_validation_tqdm(self):
        bar = tqdm(
            desc=self.validation_description,
            position=1,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
        )
        return bar


if __name__ == "__main__":
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    max_epochs = 3
    path = "F:/Factor/concat"
    label = pd.read_pickle(os.path.join(path, "label.pkl"))

    date_range = pd.date_range("2018-01-01", "2023-08-01", freq="12M").strftime(
        "%Y-%m-01"
    )

    for date in date_range:
        X_train = []
        print(date)
        period_range = pd.date_range(end=date, periods=60, freq="M").strftime(
            "%Y-%m-01"
        )
        for period in period_range:
            X_train.append(
                pd.read_pickle(os.path.join(path, "factor", f"{period}.pkl")).astype(
                    "float16"
                )
            )

        X_train = pd.concat(X_train).fillna(0)

        y_train = label.loc[X_train.index].fillna(0)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.25, random_state=random_seed
        )
        train_dataset = SectionalTrainDataset(X_train, y_train)
        valid_dataset = SectionalTrainDataset(X_valid, y_valid)
        batch_size = 256
        n_works = 1
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )

        early_stopping = EarlyStopping(monitor="val_auc", mode="max", patience=0)
        bar = LitQTDM()
        model = LitGate(
            input_dim=1447,
            n_stages=5,
            shrink=512,
        )

        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stopping],
            precision="16",
            default_root_dir=f"model/{date}",
        )
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
        )
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print(lr_finder.suggestion())
        model.hparams.lr = lr_finder.suggestion()
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )
        del train_dataloader, train_dataset, valid_dataloader, valid_dataset
        X_test = []
        test_date_range = pd.date_range(date, freq="M", periods=12).strftime("%Y-%m-01")
        for test_date in test_date_range:
            try:
                X_test.append(
                    pd.read_pickle(os.path.join(path, "factor", f"{test_date}.pkl"))
                )
            except:
                break
        X_test = pd.concat(X_test)
        test_dataset = SectionalPredictionDataset(X_test)
        test_dataloader = DataLoader(
            test_dataset, batch_size=512, num_workers=10, persistent_workers=True
        )
        score = trainer.predict(model, test_dataloader)
        score = torch.cat(score)
        score = pd.Series(score.numpy(), X_test.index).unstack()
        score.to_pickle(f"score_{date}.pkl")
        print(score)
        del test_dataloader, test_dataset, model, trainer
        break
