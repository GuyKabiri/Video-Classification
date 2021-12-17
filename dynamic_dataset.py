import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from utils import *

class DynamicFrameDataset(Dataset):

    classes = [ 'Basketball', 'Biking', 'Diving', 'PizzaTossing', 'RopeClimbing' ]

    def __init__(self, main_dir, num_frames, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.main_dir = main_dir
        self.num_frames = num_frames
        self.x = []
        self.y = []
        self.load()

    def load(self):
        for cls in os.listdir(self.main_dir):
            class_path = os.path.join(self.main_dir, cls)
            files_only = [ f for f in os.listdir(class_path) if os.path.isfile( os.path.join(class_path, f) ) ]

            for item in files_only:
                item_path = os.path.join(class_path, item)
                self.x.append(item_path)
                self.y.append(DynamicFrameDataset.classes.index(cls))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        path = self.x[idx]
        frames = []

        print(path)
        cap = cv2.VideoCapture(path)   #   create video object
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))              #   get number of frames
        frame_idx = []
        count = 0

        while count < self.num_frames:
            idx = np.random.randint(0, v_len)
            if idx in frame_idx:
                continue
            frame_idx.append(idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                if self.transforms:   
                    img = self.transforms(image=img)['image']
                frames.append(img)
                count += 1

        cap.release()
        
        frames = torch.stack(frames)
        labels = self.y[idx]

        return frames, labels




if __name__ == "__main__":
    train_path = 'data/train/'
    valid_path = 'data/valid/'

    train_dataset = DynamicFrameDataset(train_path, 16, get_transformer('valid'))
    print(len(train_dataset))
    print(len(train_dataset.x), len(train_dataset.y))

    batch = train_dataset.__getitem__(30)
    x, y = batch
    print(x[0])
    print(x.size(), y)
    # imgs, lbls = batch
    # print(imgs.shape)