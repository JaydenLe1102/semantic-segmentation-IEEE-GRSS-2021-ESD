""" Place your dataset.py code here."""
import os
import sys
import pyprojroot
sys.path.append(str(pyprojroot.here()))
from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path
from ..preprocessing.subtile_esd_hw02 import Subtile, TileMetadata
from typing import List, Dict, Tuple
from copy import deepcopy
class DSE(Dataset):
    """
    Custom dataset for the IEEE GRSS 2021 ESD dataset.

    args:
        root_dir: str | os.PathLike
            Location of the processed subtiles
        selected_bands: Dict[str, List[str]] | None
            Dictionary mapping satellite type to list of bands to select
        transform: callable, optional
            Object that applies augmentations to a sample of the data
    attributes:
        root_dir: str | os.PathLike
            Location of the processed subtiles
        tiles: List[Path]
            List of paths to the subtiles
        transform: callable
            Object that applies augmentations to the sample of the data

    """
    def __init__(self, root_dir: str | os.PathLike, selected_bands: Dict[str, List[str]] | None=None, transform=None):
        self.root_dir = Path(root_dir)
        self.tiles = list(self.root_dir.glob('*'))  # find every tiles in root_dir
        self.selected_bands = selected_bands
        self.transform = transform
        super().__init__()

    def __len__(self):
        """
            Returns number of tiles in the dataset

            Output: int
                length: number of tiles in the dataset
        """
        n_of_tiles = len(self.tiles)
        return n_of_tiles

    def __aggregate_time(self, img):
        """
            Aggregates time dimension in order to 
            feed it to the machine learning model.

            This function needs to be changed in the
            final project to better suit your needs.

            For homework 2, you will simply stack the time bands
            such that the output is shaped (time*bands, width, height),
            i.e., all the time bands are treated as a new band.

            Input:
                img: np.ndarray
                    (time, bands, width, height) array
            Output:
                new_img: np.ndarray
                    (time*bands, width, height) array
        """
        re_img = img.reshape(-1, img.shape[2], img.shape[3])
        return re_img
    
    def __select_indices(self, bands: List[str], selected_bands: List[str]):
        """
            Selects the indices of the bands used.

            Input:
                bands: List[str]
                    list of bands in the order that they are stacked in the
                    corresponding satellite stack
                selected_bands: List[str]
                    list of bands that have been selected

            Output:
                bands_indices: List[int]
                    index location of selected bands
        """
        location = []
        for v in selected_bands:
            if v in bands:
                location.append(bands.index(v))
        return location
    
    def __select_bands(self, subtile):
        """
            Aggregates time dimension in order to
            feed it to the machine learning model.

            This function needs to be changed in the
            final project to better suit your needs.

            For homework 2, you will simply stack the time bands
            such that the output is shaped (time*bands, width, height),
            i.e., all the time bands are treated as a new band.

            Input:
                subtile: Subtile object
                    (time, bands, width, height) array
            Output:
                selected_satellite_stack: Dict[str, np.ndarray]
                    satellite--> np.ndarray with shape (time, bands, width, height) array

                new_metadata: TileMetadata
                    Updated metadata with only the satellites and bands that were picked
        """
        new_metadata = deepcopy(subtile.tile_metadata)
        if self.selected_bands is not None:
            selected_satellite_stack = {}
            new_metadata.satellites = {}
            for key in self.selected_bands:
                satellite_bands = subtile.tile_metadata.satellites[key].bands
                selected_bands = self.selected_bands[key]
                indices = self.__select_indices(satellite_bands, selected_bands)
                new_metadata.satellites[key] = subtile.tile_metadata.satellites[key]
                subtile.tile_metadata.satellites[key].bands = self.selected_bands[key]
                # for i in indices:
                #   selected_satellite_stack[key][:, i, :, :] = subtile.satellite_stack[key][:, i, :, :]
                selected_satellite_stack[key] = subtile.satellite_stack[key][:, indices, :, :] # dimensions [t, bands, h, w]
        else:

            selected_satellite_stack = subtile.satellite_stack

        return selected_satellite_stack, new_metadata

    def __getitem__(
            self,
            idx: int
            ) -> tuple[np.ndarray, np.ndarray, TileMetadata]:  # changed "Tuple" to "tuple" since it keep giving syntax errors, change it back if this causing any additional errors on test
        """
            Loads subtile at index idx, then
                - selects bands
                - aggregates times
                - stacks satellites
                - performs self.transform
            
            Input:
                idx: int
                    index of subtile with respect to self.tiles
            
            Output:
                X: np.ndarray | torch.Tensor
                    input data to ML model, of shape (time*bands, width, height)
                y: np.ndarray | torch.Tensor
                    ground truth, of shape (1, width, height)
                tile_metadata:
                    corresponding tile metadata
        """
        # load the subtiles using the Subtile class in
        # src/preprocessing/subtile_esd_hw02.py
                
        # call the __select_bands function to select the bands and satellites

        # stack the time dimension with the bands, this will treat the
        # timestamps as bands for the model you may want to change this
        # depending on your model and depending on which timestamps and
        # bands you want to use

        # Concatenate the time and bands
        
        # Adjust the y ground truth to be the same shape as the X data by
        # removing the time dimension

        # all timestamps are treated and stacked as bands

        # if there is a transform, apply it to both X and y
        subtile_path_or_dir = self.tiles[idx]
        
        all_satellite_data = []
        all_gt_data = []
        all_metadata = []

        def process_subtile(subtile_path):
            subtile = Subtile().load(subtile_path)
            selected_satellite_stack, new_metadata = self.__select_bands(subtile)
            stacked_satellite_data = []
            
            if 'gt' not in selected_satellite_stack and 'gt' in subtile.satellite_stack:
                selected_satellite_stack['gt'] = subtile.satellite_stack['gt']
            
            for satellite_type, stack in selected_satellite_stack.items():
                if satellite_type != 'gt':
                    aggregated_data = self.__aggregate_time(stack)
                    stacked_satellite_data.append(aggregated_data)
                else:
                    all_gt_data.append(stack)
            all_metadata.append(new_metadata)
            if stacked_satellite_data:
                X = np.concatenate(stacked_satellite_data, axis=0)
                all_satellite_data.append(X)
            else:
                raise ValueError("No satellite data found for selected bands.")
        
        if os.path.isdir(subtile_path_or_dir):
            for file_name in os.listdir(subtile_path_or_dir):
                if file_name.endswith('.npy') or file_name.endswith('.npz'):
                    full_path = os.path.join(subtile_path_or_dir, file_name)
                    process_subtile(full_path)
        elif os.path.isfile(subtile_path_or_dir):
            process_subtile(subtile_path_or_dir)
        else:
            raise ValueError(f"The path {subtile_path_or_dir} is neither a file nor a directory.")
        
        if not all_satellite_data:
            raise ValueError("No satellite data found.")
        X_combined = np.concatenate(all_satellite_data, axis=0)
        
        if not all_gt_data:
           raise ValueError("No ground truth data found.")
        y_combined = np.concatenate(all_gt_data, axis=0)
        y_combined = y_combined[:, 0, :, :]

        if self.transform:
            transformed  = self.transform({'X': X_combined, 'y': y_combined})
            X_combined = transformed.get('X')
            y_combined = transformed.get('y')
        
        if not isinstance(X_combined, torch.Tensor):
            X_combined = torch.from_numpy(X_combined).float()
            
        if not isinstance(y_combined, torch.Tensor):
            y_combined = torch.from_numpy(y_combined).float()
        
        if all_metadata:
            tile_metadata = all_metadata[0]
        else:
            raise ValueError("No metadata found.")
        
        y_combined = torch.subtract(y_combined, 1.0)

        return X_combined, y_combined, tile_metadata
    
        