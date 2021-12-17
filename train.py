from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

from utils import get_loaders
from utils import *
from model import *
from config import *

def train(config):
    loaders = get_loaders(config.batch_size, config.num_workers)

    wandb_logger = WandbLogger(project="video-classification")
    model = LitFrames(drop_prob=config.drop_prob, num_frames=config.num_frames, num_classes=config.num_classes)

    trainer = Trainer(
        gpus=1,
        logger=wandb_logger,
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0
    )
    trainer.fit(model, loaders['train'], loaders['valid'])
    

if __name__ == "__main__":
    seed_everything(CFG.seed, workers=True)
    train(CFG)