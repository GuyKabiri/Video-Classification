import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics.functional import accuracy
import torchvision.models as models
from dataset import *
import pytorchvideo.models.slowfast as SlowFastModel

class SlowFastLitFrames(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=16, num_classes=5):
        super().__init__()

        self.drop_prob = drop_prob
        self.num_classes = num_classes
        self.num_frames = num_frames

        self.load()

        

    def load(self):
        self.backbone = SlowFastModel.create_slowfast(
            model_num_class=self.num_classes,
            dropout_rate=self.drop_prob,
        )

    def forward(self, x):
        # batch_size, n_frames, n_channels, height, width = x.size()  #   shape is [ batch, frames, 3, height, width ]

        out = self.backbone(x)

        return out


    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.learning_rate, weight_decay=1e-4)
        # optimizer = torch.optim.SGD(self.parameters(), lr=CFG.learning_rate, momentum=0.9, weight_decay=1e-3)
        optimizer = torch.optim.ASGD(self.parameters(), lr=CFG.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=4, cooldown=2)
        return { "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss" }

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["valid_loss"])
        else:
            sch.step()
    
    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        output = self(x)

        acc = accuracy(output, y)
        loss = F.cross_entropy(output, y)
        metrics = {"train_acc": acc, "train_loss": loss}

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"valid_acc": acc, "valid_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        output = self(x)

        acc = accuracy(output, y)
        loss = F.cross_entropy(output, y)

        return loss, acc