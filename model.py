import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics.functional import accuracy
import torchvision.models as models
from dataset import *

class LitFrames(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=16, num_classes=5):
        super().__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames

        self.backbone = models.resnet50(pretrained=True)
        
        out_channels = self.backbone.conv1.out_channels
        in_features = self.backbone.fc.in_features
        self.backbone.conv1 = nn.Conv2d(3*num_frames, out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))  #   changing 1st conv layer to read 3xnum_frames for early fusion
        self.backbone.fc = nn.Identity()    #    y(x)=x
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features, num_classes)


    def forward(self, x):
        batch_size, n_frames, n_channels, height, width = x.size()  #   shape is [ batch, frames, 3, height, width ]

        x = x.view(batch_size, n_frames*n_channels, height, width)  #   convert shape to [ batch, 3 x frames, height, width ]
        out = self.backbone(x)
        out = self.dropout(self.relu(out))
        out = self.fc(out)

        return out


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=CFG.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2)
        return { "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss" }

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["valid_loss"])
        else:
            sch.step()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
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
        x, y = batch
        output = self(x)

        acc = accuracy(output, y)
        loss = F.cross_entropy(output, y)

        return loss, acc