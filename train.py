from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything

from utils import get_loader
from utils import *
from model import *
from config import *

def train(config):

    loaders = {
        p: get_loader(p, config.batch_size, config.num_workers, config.dynamic_frames, config.num_frames)
            for p in [ 'train', 'valid', 'test'] 
    }

    wandb_logger = WandbLogger(project='video-classification')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = LitFrames(drop_prob=config.drop_prob, num_frames=config.num_frames, num_classes=config.num_classes)

    trainer = Trainer(
        gpus=1,
        logger=wandb_logger,
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0,
        # overfit_batches=0.05,
        callbacks=[lr_monitor],
    )
    trainer.fit(model, loaders['train'], loaders['valid'])
    trainer.test(model, loaders['test'])
    

if __name__ == "__main__":
    seed_everything(CFG.seed, workers=True)
    train(CFG)