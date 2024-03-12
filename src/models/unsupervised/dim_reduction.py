import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from typing import Tuple, Any
from src.esd_data.datamodule import ESDDataModule

def preprocess_for_dim_reduction(esd_datamodule: ESDDataModule) -> (np.ndarray, np.ndarray):
    """
    Preprocess the data for the dimensionality reduction

    Input: 
        esd_datamodule: ESDDataModule
            datamodule to load the data from

    Output:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

        y_flat: np.ndarray
            Flattened ground truth of shape (sample, 1)
    """

    # load the data from the datamodule
    train_data = esd_datamodule.train_dataloader()
    
    # concatenate the data
    con_X = []
    con_Y = []

    for x, y, m in train_data:
        con_X.append(x)
        con_Y.append(y)

    coned_X = np.concatenate(con_X, axis=0)
    coned_Y = np.concatenate(con_Y, axis=0)
    
    # flatten the data
    X_flat = coned_X.reshape(coned_X.shape[0], -1)
    Y_flat = coned_Y.reshape(-1, 1)

    return X_flat, Y_flat
    

def perform_PCA(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    PCA is commonly used for dimensionality reduction by projecting each data
    point onto only the first few principal components to obtain
    lower-dimensional data while preserving as much of the data's variation
    as possible.

    For more information:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Input:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        pca: PCA
            PCA object
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_flat)
    return X_pca, pca


def perform_TSNE(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, TSNE]:
    """
    t-SNE (t-distributed Stochastic Neighbor Embedding) is an unsupervised
    non-linear dimensionality reduction technique for data exploration
    and visualizing high-dimensional data. Non-linear dimensionality
    reduction means that the algorithm allows us to separate data that
    cannot be separated by a straight line.

    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Input:
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        tsne: TSNE
            TSNE object
    """
    tsne = TSNE(n_components=n_components)
    X_tsne = tsne.fit_transform(X_flat)
    return X_tsne, tsne

def perform_UMAP(X_flat: np.ndarray, n_components: int) -> Tuple[np.ndarray, UMAP]:
    """
    UMAP stands for Uniform Manifold Approximation and Projection.
    It is a dimension reduction technique that helps in visualizing
    high-dimensional data.

    https://umap-learn.readthedocs.io/en/latest/

    Input: 
        X_flat: np.ndarray
            Flattened tile of shape (sample,time*band*width*height)

    Output:
        X_pca: np.ndarray
            Dimensionality reduced array of shape (sample, n_components)
        umap: UMAP
            UMAP object 
    """
    umap = UMAP(n_components=n_components)
    X_umap = umap.fit_transform(X_flat)
    return X_umap, umap
    
