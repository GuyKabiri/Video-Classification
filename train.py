

from utils import get_loader
from utils import *
from model import *
from config import *

def train(config):
    loaders = get_loaders(config.batch_size, config.num_workers)

    model = LitFrames(drop_prob=config.drop_prob, num_frames=config.num_frames, num_classes=config.num_classes)
    

if __name__ == "__main__":
    seed_everything(CFG.seed)
    train(CFG)