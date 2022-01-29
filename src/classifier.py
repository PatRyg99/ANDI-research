import os

import numpy as np
import pytorch_lightning as pl

import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim
import torch.utils
import torch.utils.data
from torch.nn import CrossEntropyLoss
import torchmetrics

import monai
from monai.data import list_data_collate
from monai.utils import set_determinism
from monai.transforms import Compose

from src.transforms import ToFloatTensord, Normalized, RandomNormalNoised
from src.models.trajectorynet import TrajectoryNet
from configs.train_config import TrainClassifierConfig


class TrajectoryClassifier(pl.LightningModule):
    def __init__(self, config: TrainClassifierConfig):
        super().__init__()

        self.save_hyperparameters()
        set_determinism(seed=0)

        self.dataset_path = config.dataset_path
        self.one_hotted = config.one_hotted
        self.bs = config.bs
        self.lr = config.lr

        self.trajectory_key = "trajectory"
        self.label_key = "label"
        self.alpha_key = "alpha"
        self.loss = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=config.classes, average="macro")
        self.f1_score = torchmetrics.F1Score(
            num_classes=config.classes, average="macro"
        )

        self.model = TrajectoryNet(
            config.dim,
            config.channels,
            config.classes,
            config.kernel_size,
            config.stride,
        )

    def forward(self, x):
        return self.model(x)

    def remap_labels(self, labels):
        uniques = np.unique(labels)

        for unique in uniques:
            labels[labels == unique] = np.argwhere(uniques == unique).flatten()

        return labels

    def prepare_data(self):

        # Load train data
        train_X = np.load(os.path.join(self.dataset_path, "X_train.npy"))
        train_y = np.load(os.path.join(self.dataset_path, "y_train.npy"))

        if self.one_hotted:
            train_y = np.argmax(train_y, axis=1)

        # Load valid data
        val_X = np.load(os.path.join(self.dataset_path, "X_val.npy"))
        val_y = np.load(os.path.join(self.dataset_path, "y_val.npy"))

        if self.one_hotted:
            val_y = np.argmax(val_y, axis=1)

        # Prepare dictionaries
        train_dict = [
            {self.trajectory_key: X, self.label_key: y}
            for X, y in zip(train_X, train_y)
        ]

        val_dict = [
            {self.trajectory_key: X, self.label_key: y} for X, y in zip(val_X, val_y)
        ]

        # Transforms
        train_tfms = Compose(
            [
                Normalized(keys=[self.trajectory_key]),
                RandomNormalNoised(keys=[self.trajectory_key]),
                ToFloatTensord(keys=[self.trajectory_key, self.label_key]),
            ]
        )
        val_tfms = Compose(
            [
                Normalized(keys=[self.trajectory_key]),
                ToFloatTensord(keys=[self.trajectory_key, self.label_key]),
            ]
        )

        self.train_ds = monai.data.CacheDataset(
            data=train_dict, cache_rate=1.0, transform=train_tfms, num_workers=6,
        )

        self.val_ds = monai.data.CacheDataset(
            data=val_dict, cache_rate=1.0, transform=val_tfms, num_workers=6
        )

    def train_dataloader(self):
        train_loader = monai.data.DataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=3,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = monai.data.DataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=3,
            collate_fn=list_data_collate,
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        return optimizer

    def shared_step(self, batch):

        trajectories, labels = batch[self.trajectory_key], batch[self.label_key]
        preds = self.forward(trajectories)

        loss = self.loss(preds, labels.long())

        log_preds = F.log_softmax(preds)
        acc = self.acc(log_preds, labels.int())
        f1_score = self.f1_score(log_preds, labels.int())

        return loss, acc, f1_score

    def training_step(self, batch, batch_idx):
        loss, acc, f1_score = self.shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_f1_score", f1_score, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1_score = self.shared_step(batch)

        metrics = {"val_loss": loss, "val_acc": acc, "val_f1_score": f1_score}
        self.log_dict(metrics, on_epoch=True)
        return metrics
