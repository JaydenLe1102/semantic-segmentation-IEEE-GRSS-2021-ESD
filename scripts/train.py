import pyprojroot
import sys
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)


from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation

import wandb
@dataclass
class ESDConfig:
    """
    IMPORTANT: This class is used to define the configuration for the experiment
    Please make sure to use the correct types and default values for the parameters
    and that the path for processed_dir contain the tiles you would like 
    """
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    selected_bands: None = None
    model_type: str = "SegmentationCNN"
    tile_size_gt: int = 4
    batch_size: int = 8
    max_epochs: int = 2
    seed: int = 12378921
    learning_rate: float = 1e-3
    num_workers: int = 11
    accelerator: str = "gpu"
    devices: int = 1
    in_channels: int = 99
    out_channels: int = 4
    depth: int = 2
    n_encoders: int = 4
    embedding_size: int = 64
    pool_sizes: str = "5,5,2" # List[int] = [5,5,2]
    kernel_size: int = 3
    scale_factor: int = 50
    wandb_run_name: str | None = None

# not sure if we can use that
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger


def train(options: ESDConfig):
    """
    Prepares datamodule and model, then runs the training loop

    Inputs:
        options: ESDConfig
            options for the experiment
    """
    # Initialize the weights and biases logger
    if options.wandb_run_name: #make sure name is not none
        wb_logger = WandbLogger(name=options.wandb_run_name, log_model="all")
    else:
        #wb_logger = WandbLogger(log_model="all")
        wb_logger = None
        
        
    csv_logger = CSVLogger(save_dir=root / 'logs', name=options.model_type)
    

    # initiate the ESDDatamodule
    # use the options object to initiate the datamodule correctly
    # make sure to prepare_data in case the data has not been preprocessed
    esd = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        selected_bands=options.selected_bands,
        tile_size_gt=options.tile_size_gt,
        batch_size=options.batch_size,
        num_workers=options.num_workers
        #seed=options.seed       
    )
    esd.prepare_data()
    esd.setup("train")
    
    # create a dictionary with the parameters to pass to the models
    params_dict = {       
        "depth": options.depth,
        "embedding_size": options.embedding_size,
        "pool_sizes": [int(v) for v in options.pool_sizes.split(',')],
        "kernel_size": options.kernel_size,
        "scale_factor": options.scale_factor,
    }

    # initialize the ESDSegmentation module
    model = ESDSegmentation(
        model_type=options.model_type,
        in_channels=99,
        out_channels=4,
        learning_rate=options.learning_rate,
        model_params=params_dict,
        )

    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath=root / 'models' / options.model_type,
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
            save_top_k=0,
            save_last=True,
            verbose=True,
            monitor='val_loss',
            mode='min',
            every_n_train_steps=1000
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    # create a pytorch Trainer
    # see pytorch_lightning.Trainer
    # make sure to use the options object to load it with the correct options
    trainer = pl.Trainer(
        num_nodes=1,
        max_epochs=options.max_epochs,
        accelerator=options.accelerator,
        devices=options.devices,
        callbacks=callbacks,
        logger = wb_logger,
        # some optional params that might helpful for debugging, uncomment when needed
        # IMPORTANT: make sure to add "," for logger before you uncomment

        # this would control how open our model take logs, default is 50
        # log_every_n_steps=50,

        # this for debug, so that our model always product same result for same setup, default is false
        # WARNING: may severly extend training time
        # deterministic=True
    )

    # run trainer.fit
    # make sure to use the datamodule option
    trainer.fit(model, datamodule=esd)



if __name__ == '__main__':
    # load dataclass arguments from yml file
    
    config = ESDConfig()
    parser = ArgumentParser()

    
    parser.add_argument("--model_type", type=str, help="The model to initialize.", default=config.model_type)
    parser.add_argument("--learning_rate", type=float, help="The learning rate for training model", default=config.learning_rate)
    parser.add_argument("--max_epochs", type=int, help="Number of epochs to train for.", default=config.max_epochs)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    
    parser.add_argument('--in_channels', type=int, default=config.in_channels, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=config.out_channels, help='Number of output channels')
    parser.add_argument('--depth', type=int, help="Depth of the encoders (CNN only)", default=config.depth)
    parser.add_argument('--n_encoders', type=int, help="Number of encoders (Unet only)", default=config.n_encoders)
    parser.add_argument('--embedding_size', type=int, help="Embedding size of the neural network (CNN/Unet)", default=config.embedding_size)
    parser.add_argument('--pool_sizes', help="A comma separated list of pool_sizes (CNN only)", type=str, default=config.pool_sizes)
    parser.add_argument('--kernel_size', help="Kernel size of the convolutions", type=int, default=config.kernel_size)
    parser.add_argument('--scale_factor', help="Scale factor between the labels and the image (Unet and Transfer Resnet)", type=int, default=config.scale_factor)
    # --pool_sizes=5,5,2 to call it correctly
    
    parse_args = parser.parse_args()
    
    train(ESDConfig(**parse_args.__dict__))
