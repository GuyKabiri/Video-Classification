import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from utils import *

class FrameDataset(Dataset):

    classes = [ 'Basketball', 'Biking', 'Diving', 'PizzaTossing', 'RopeClimbing' ]

    def __init__(self, main_dir, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.main_dir = main_dir
        self.x = []
        self.y = []
        self.load()

    def load(self):
        for cls in os.listdir(self.main_dir):
            class_path = os.path.join(self.main_dir, cls)
            folders_only = [ f for f in os.listdir(class_path) if os.path.isdir( os.path.join(class_path, f) ) ]

            for item in folders_only:
                item_path = os.path.join(class_path, item)
                frames = os.listdir(item_path)
                frames = [ os.path.join(item_path, f) for f in frames ]
                self.x.append(frames)
                self.y.append(FrameDataset.classes.index(cls))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        frame_paths = self.x[idx]
        frames = []

        for path in frame_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            if self.transforms:   
                img = self.transforms(image=img)['image']
            frames.append(img)

        
        frames = torch.stack(frames)
        labels = self.y[idx]

        return frames, labels




if __name__ == "__main__":
    train_path = 'data/train/'
    valid_path = 'data/valid/'

    train_dataset = FrameDataset(train_path, get_transformer('valid'))
    print(len(train_dataset))
    print(len(train_dataset.x), len(train_dataset.y))

    batch = train_dataset.__getitem__(30)
    # imgs, lbls = batch
    # print(imgs.shape)