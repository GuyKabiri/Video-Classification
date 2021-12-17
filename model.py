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

        self.backbone = models.resnet101(pretrained=True)
        
        out_channels = self.backbone.conv1.out_channels
        in_features = self.backbone.fc.in_features
        self.backbone.conv1 = nn.Conv2d(3*num_frames, out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.backbone.fc = nn.Identity()    #    y(x)=x
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features, num_classes)


    def forward(self, x):
        batch_size, n_frames, n_channels, height, width = x.size()

        x = x.view(batch_size, n_frames*n_channels, height, width)
        out = self.backbone(x)
        out = self.dropout(self.relu(out))
        out = self.fc(out)

        return out


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
    def training_step(self, batch, b_idx):
        x, y = batch

        output = self(x)
        pred = torch.argmax(output, dim=1)

        accur = accuracy(pred, y)
        loss = F.cross_entropy(output, y)

        self.log("train_loss", loss)
        self.log("train_accuracy", accur)
        
        return loss

    def validation_step(self, batch, b_idx):
        x, y = batch

        output = self(x)
        pred = torch.argmax(output, dim=1)

        accur = accuracy(pred, y)
        loss = F.cross_entropy(output, y)

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accur)
        
        return loss