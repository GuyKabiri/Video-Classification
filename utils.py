import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *
from dataset import FrameDataset

    
def get_transformer(phase):
    valid_trans = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(p=1.0)
    ])

    if phase == 'train':
        return A.Compose([
            A.OneOf([
                # A.ChannelDropout(p=0.5),
                A.Emboss(p=0.5),
                A.Sharpen(p=0.5),
            ], p=0.5),
            A.Rotate(p=0.5, limit=[-35, 35]),
            A.MotionBlur(p=0.3),
            valid_trans,
        ])

    return valid_trans


def get_loader(phase, batch_size=4, num_workers=8):
    phases = [ 'train', 'valid' ]

    paths = { 
        p: 'data/{}'.format(p)
                for p in phases
    }

    datasets = {
        p:  FrameDataset(paths[p], get_transformer(p))
                for p in phases
    }

    return {
        p: DataLoader(
            dataset=datasets[p],
            batch_size=batch_size,
            shuffle=p=='train',
            num_workers=num_workers
        )
                for p in phases
    }




def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True