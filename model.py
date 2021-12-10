import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import torchvision.models as models
from dataset import *

class LitFrames(LightningModule):
    def __init__(self, drop_rate=0.5, num_frames=16, num_classes=5):
        super().__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.backbone = models.resnet101(pretrained=True)
        self.num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(num_frames*self.num_features , num_classes)

    def forward(self, x):
        batch_size, n_frames, n_channels, height, width = x.size()

        # shape = [ batch, frames, channels, height, width ]
        frame_idx = 1

        out = torch.Tensor( (self.num_frames*self.num_features) )

        for i in range(x.shape[ frame_idx ]):
            tmp = self.backbone(x[ :, i, :, : ])
            print(tmp.size())
            start_idx = i * self.num_features
            end_idx = start_idx + self.num_features
            out[ start_idx : end_idx ] = tmp
            del tmp
        
        out = self.dropout(out)
        out = self.fc1(out)

        return out