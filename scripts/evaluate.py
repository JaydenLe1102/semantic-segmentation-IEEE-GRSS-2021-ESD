import pyprojroot
import sys
import os
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import (
    restitch_eval,
    restitch_and_plot
)
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import tifffile
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_default_device("mps")

@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    results_dir: str | os.PathLike = root / 'data/predictions' / "UNet"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "UNet" / "last.ckpt"



def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    #raise NotImplementedError # Complete this function using the code snippets below. Do not forget to remove this line.
    # Load datamodule
    
    dm = ESDDataModule(options.processed_dir, options.raw_dir, options.selected_bands, options.tile_size_gt, options.batch_size, options.seed)
    dm.prepare_data()
    dm.setup("fit")
    
    # load model from checkpoint at options.model_path
    
    model = ESDSegmentation.load_from_checkpoint(options.model_path)

    # set the model to evaluation mode (model.eval())
    model.eval()
    
    # this is important because if you don't do this, some layers
    # will not evaluate properly

    # instantiate pytorch lightning trainer
    #plTrainer = pl.Trainer()

    # run the validation loop with trainer.validate
    #plTrainer.validate(model = model, datamodule=dm)

    # run restitch_and_plot

    # for every subtile in options.processed_dir/Val/subtiles
    
    # run restitch_eval on that tile followed by picking the best scoring class
    # save the file as a tiff using tifffile
    # save the file as a png using matplotlib
    tiles = set([extract_tile_id(subtile) for subtile in os.listdir(options.processed_dir / "Val" / "subtiles")])
    print(sorted(tiles))
    for parent_tile_id in sorted(tiles):
        
        restitch_and_plot(options = options, datamodule= dm, model= model, parent_tile_id= parent_tile_id, image_dir = options.results_dir)

        stitched_image,stitched_ground_truth, y_pred  = restitch_eval(options.processed_dir / "Val", "sentinel2", parent_tile_id, (0, 4), (0, 4), dm, model)
        
        y_pred = np.argmax(y_pred, axis=0)
        
        tifffile.imsave(options.results_dir / f"{parent_tile_id}.tiff", stitched_ground_truth[0])
        
        # freebie: plots the predicted image as a jpeg with the correct colors
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(stitched_ground_truth[0], vmin=-0.5, vmax=3.5,cmap=cmap)
        plt.savefig(options.results_dir / f"{parent_tile_id}.png")
        

def extract_tile_id(filename):
  underscore_index = filename.find("_")
  if underscore_index == -1:
    return None  # No underscore found, not a valid tile name format

  return filename[:underscore_index]
    

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results dir")
    main(EvalConfig(**parser.parse_args().__dict__))