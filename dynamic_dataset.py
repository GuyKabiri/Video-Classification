import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from utils import *
from PackPathwayTransform import PackPathway

class DynamicFrameDataset(Dataset):

    classes = [ 'Basketball', 'Biking', 'Diving', 'PizzaTossing', 'RopeClimbing' ]

    def __init__(self, main_dir, num_frames, transforms=None, slowfast=False):
        super().__init__()
        self.transforms = transforms
        self.main_dir = main_dir
        self.num_frames = num_frames
        self.slowfast = slowfast
        self.pack_pathway = PackPathway()
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

        cap = cv2.VideoCapture(path)   #   open video
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))              #   get number of frames
        frame_idx = np.sort(np.random.choice(np.arange(v_len-1), self.num_frames))   #   get random frame indices, sometimes the last frame generates an error, therefore v_len-1

        #   iterate for each frame
        for i in frame_idx:
            img = torch.zeros((3, 224, 224))    #   empty tensor in case frame will not read by cv2
            cap.set(cv2.CAP_PROP_POS_FRAMES, i) #   move to relevant frame index
            ret, frame = cap.read()

            # if frame was read
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                if self.transforms:   
                    img = self.transforms(image=img)['image']
            frames.append(img)

        cap.release()   #   release video
        
        labels = self.y[idx]
        frames = torch.stack(frames)

        if self.slowfast:
            # change view to [ C, F, H, W ]
            frames = torch.permute(frames, (1, 0, 2, 3))
            frames = self.pack_pathway(frames)
            return frames, labels, idx, dict()

        return frames, labels




if __name__ == "__main__":
    train_path = 'data/train/'
    valid_path = 'data/valid/'

    train_dataset = DynamicFrameDataset(valid_path, 16, get_transformer('valid'))
    print(len(train_dataset))
    print(len(train_dataset.x), len(train_dataset.y))

    batch = train_dataset.__getitem__(30)
    x, y = batch
    # print(x[0])
    print(x.size(), y)
    print(len(train_dataset.x), len(train_dataset.y))

    loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle= True,
        num_workers=8
    )

    batch = next(iter(loader))
    # imgs, lbls = batch
    # print(imgs.shape)