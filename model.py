import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import torchvision.models as models
from dataset import *

class LitFrames(LightningModule):
    def __init__(self, drop_prob=0.5, num_frames=16, num_classes=5):
        super().__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.backbone = models.resnet101(pretrained=True)

    def forward(self, x):
        batch_size, n_frames, n_channels, height, width = x.size()

        out = self.backbone(x)

        return out