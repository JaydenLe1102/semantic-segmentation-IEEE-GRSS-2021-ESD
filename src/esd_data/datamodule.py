""" place your datamodule code here. """
""" This module contains the PyTorch Lightning ESDDataModule to use with the
PyTorch ESD dataset."""

import pytorch_lightning as pl
from torch import Generator
from torch.utils.data import DataLoader, random_split
import torch
from .dataset import DSE
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ..preprocessing.subtile_esd_hw02 import grid_slice
from ..preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_viirs,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
)
from ..preprocessing.file_utils import (
    load_satellite
)
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor
)
from torchvision import transforms
from copy import deepcopy
from typing import List, Tuple, Dict
from src.preprocessing.file_utils import Metadata
import random

def collate_fn(batch):
    Xs = []
    ys = []
    metadatas = []
    for X, y, metadata in batch:
        Xs.append(X)
        ys.append(y)
        metadatas.append(metadata)

    Xs_float32 = [X.float() for X in Xs]  # Convert each X to float32 tensor
    ys_float32 = [y.float() for y in ys]  # Convert each y to float32 tensor (assuming it's a tensor)

    return torch.stack(Xs_float32), torch.stack(ys_float32), metadatas

class ESDDataModule(pl.LightningDataModule):
    """
        PyTorch Lightning ESDDataModule to use with the PyTorch ESD dataset.

        Attributes:
            processed_dir: str | os.PathLike
                Location of the processed data
            raw_dir: str | os.PathLike
                Location of the raw data
            selected_bands: Dict[str, List[str]] | None
                Dictionary mapping satellite type to list of bands to select
            tile_size_gt: int
                Size of the ground truth tiles
            batch_size: int
                Batch size
            seed: int
                Seed for the random number generator
    """
    def __init__(
            self,
            processed_dir: str | os.PathLike,
            raw_dir: str | os.PathLike,
            selected_bands: Dict[str, List[str]] | None = None,
            tile_size_gt=4,
            batch_size=32,
            num_workers=11,
            seed=12378921):
        
        
        super().__init__()
            
        
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir_train = processed_dir / "Train"
        self.processed_dir_val = processed_dir / "Val"
        self.seed = seed
        self.batch_size = batch_size
        self.tile_size_gt = tile_size_gt
        self.selected_bands = selected_bands
        self.num_workers = num_workers
        
        rand_apply = transforms.RandomApply([AddNoise(), Blur(), RandomHFlip(1.0), RandomVFlip(1.0)], p=0.5)

        self.transform = transforms.Compose([rand_apply,ToTensor()])
                
        # set transform to a composition of the following transforms:
        # AddNoise, Blur, RandomHFlip, RandomVFlip, ToTensor
        # utilize the RandomApply transform to apply each of the transforms 
        # with a probability of 0.5
       
    
    def __load_and_preprocess(
            self,
            tile_dir: str | os.PathLike,
            satellite_types: List[str] = ["viirs", "sentinel1", "sentinel2", "landsat", "gt"]
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Metadata]]]:
        """
            Performs the preprocessing step: for a given tile located in tile_dir,
            loads the tif files and preprocesses them just like in homework 1.

            Input:
                tile_dir: str | os.PathLike
                    Location of raw tile data
                satellite_types: List[str]
                    List of satellite types to process

            Output:
                satellite_stack: Dict[str, np.ndarray]
                    Dictionary mapping satellite_type -> (time, band, width, height) array
                satellite_metadata: Dict[str, List[Metadata]]
                    Metadata accompanying the statellite_stack
        """
        preprocess_functions = {
            "viirs": preprocess_viirs,
            "sentinel1": preprocess_sentinel1,
            "sentinel2": preprocess_sentinel2,
            "landsat": preprocess_landsat,
            "gt": lambda x: x
        }
        
        satellite_stack = {}
        satellite_metadata = {}
        for satellite_type in satellite_types:
            stack, metadata = load_satellite(tile_dir, satellite_type)

            stack = preprocess_functions[satellite_type](stack)

            satellite_stack[satellite_type] = stack.astype(np.float32)
            satellite_metadata[satellite_type] = metadata

        satellite_stack["viirs_maxproj"] = np.expand_dims(maxprojection_viirs(satellite_stack["viirs"], clip_quantile=0.0), axis=0)
        satellite_metadata["viirs_maxproj"] = deepcopy(satellite_metadata["viirs"])
        for metadata in satellite_metadata["viirs_maxproj"]:
            metadata.satellite_type = "viirs_maxproj"
        
        
        return satellite_stack, satellite_metadata
    
    def _split_list_by_ratio(self, data, ratio):
        """Splits a list into two lists based on a given ratio.

        Args:
            data: The list to be split.
            ratio: The desired ratio for one list (e.g., 0.7 for 70/30 split).

        Returns:
            A tuple containing the two split lists.
        """
        random.shuffle(data)
        split_point = int(len(data) * ratio)
        return data[:split_point], data[split_point:]
    
    def _load_preprocess_save(self, data_list_path, processed_dir):
        """Loads, preprocesses, and saves the data in the given list of paths.

        Args:
            data_list_path: The list of paths to the data to be loaded and saved.
        """
        for parent_image in data_list_path:
            
            parent_image_dir = os.path.join(self.raw_dir, parent_image)
            
            satellite_stack, satellite_metadata = self.__load_and_preprocess(parent_image_dir)

            subtiles = grid_slice(satellite_stack, satellite_metadata, self.tile_size_gt)
            
            for subtiles_data in subtiles:
                subtiles_data.save(processed_dir)


    def prepare_data(self):
        """
            If the data has not been processed before (denoted by whether or not self.processed_dir is an existing directory)

            For each tile,
                - load and preprocess the data in the tile
                - grid slice the data
                - for each resulting subtile
                    - save the subtile data to self.processed_dir
        """
        # if the processed_dir does not exist, process the data and create
        # subtiles of the parent image to save
        
            # fetch all the parent images in the raw_dir
            
            # for each parent image in the raw_dir
            
                # call __load_and_preprocess to load and preprocess the data for all satellite types

                # grid slice the data with the given tile_size_gt

                # save each subtile
                

        if not Path(self.processed_dir_train).exists() and not Path(self.processed_dir_val).exists():
            os.makedirs(self.processed_dir_train)
            os.makedirs(self.processed_dir_val)
            
            train_data_list, test_data_list = self._split_list_by_ratio(list(Path(self.raw_dir).glob('*')), 0.8)
            
            self._load_preprocess_save(train_data_list,self.processed_dir_train)
            self._load_preprocess_save(test_data_list, self.processed_dir_val)
        else:
            print("Data already prepared")

                
    
    def setup(self, stage: str):
        """
            Create self.train_dataset and self.val_dataset.0000ff

            Hint: Use torch.utils.data.random_split to split the Train
            directory loaded into the PyTorch dataset DSE into an 80% training
            and 20% validation set. Set the seed to 1024.
        """

        self.train_dataset = DSE(self.processed_dir_train / 'subtiles', self.selected_bands, transform=self.transform)
        self.val_dataset = DSE(self.processed_dir_val / 'subtiles', self.selected_bands, transform=self.transform)
        
        self.stage = stage
            
            
    def train_dataloader(self):
        """
            Create and return a torch.utils.data.DataLoader with
            self.train_dataset
        """

        train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            #num_workers=self.num_workers,
            collate_fn=collate_fn,
            #persistent_workers=True
        )
        
        return train_dataloader

    
    def val_dataloader(self):
        """
            Create and return a torch.utils.data.DataLoader with
            self.val_dataset
        """

        val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            #num_workers=self.num_workers,
            collate_fn=collate_fn,
            #persistent_workers=True  
        )

        return val_dataloader
