""" This module contains functions for plotting satellite images. """
import os
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from ..preprocessing.file_utils import Metadata
from ..preprocessing.preprocess_sat import minmax_scale
from ..preprocessing.preprocess_sat import (
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
    preprocess_viirs
)


def plot_viirs_histogram(
        viirs_stack: np.ndarray,
        image_dir: None | str | os.PathLike = None,
        n_bins=100
        ) -> None:
    """
    This function plots the histogram over all VIIRS values.
    note: viirs_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """

    flat_v = viirs_stack.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(flat_v, bins=n_bins, color='blue', alpha=0.7, log=True)
    plt.title('Histogram of VIIRS (Log base)')

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "VIIRS_histogram.png")
        plt.close()




def plot_sentinel1_histogram(
        sentinel1_stack: np.ndarray,
        metadata: List[Metadata],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the Sentinel-1 histogram over all Sentinel-1 values.
    note: sentinel1_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    sentinel1_stack : np.ndarray
        The Sentinel-1 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """

    band = sentinel1_stack.shape[2]
    fig, axes = plt.subplots(nrows=band, ncols=2)

    for i in range(band):
        flattened = sentinel1_stack[:, :, i, :, :].flatten()

        ax1 = axes[i][0]
        ax1.hist(flattened, bins=n_bins, log=True)
        try:
            ax1.set_title(f'{metadata[0][0].bands[i]} bands (Log Scale)')
        except TypeError:
            ax1.set_title(f'{metadata[0].bands[i]} bands (Log Scale)')

        ax2 = axes[i][1]
        ax2.hist(flattened, bins=n_bins)
        try:
            ax2.set_title(f'{metadata[0][0].bands[i]} bands')
        except TypeError:
            ax2.set_title(f'{metadata[0].bands[i]} bands')

    plt.tight_layout()

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel1_histogram.png")
        plt.close()


def plot_sentinel2_histogram(
        sentinel2_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20) -> None:
    """
    This function plots the Sentinel-2 histogram over all Sentinel-2 values.

    Parameters
    ----------
    sentinel2_stack : np.ndarray
        The Sentinel-2 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-2 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    bands = sentinel2_stack.shape[2]
    fig, axes = plt.subplots(nrows=bands // 2, ncols=2)
    axes = axes.ravel()

    for i in range(bands):
        flattened = sentinel2_stack[:, :, i, :, :].flatten()

        ax = axes[i]
        ax.hist(flattened, bins=n_bins, log=True)
        try:
            ax.set_title(f'{metadata[0][0].bands[i]} bands (Log Scale)')
        except TypeError:
            ax.set_title(f'{metadata[0].bands[i]} bands (Log Scale)')

    plt.tight_layout()

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel2_histogram.png")
        plt.close()


def plot_landsat_histogram(
        landsat_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the landsat histogram over all landsat values over all
    tiles present in the landsat_stack.

    Parameters
    ----------
    landsat_stack : np.ndarray
        The landsat image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the landsat image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    bands = landsat_stack.shape[2]
    fig, axes = plt.subplots(nrows=bands // 2, ncols=3)
    axes = axes.ravel()

    for i in range(bands):
        flattened = landsat_stack[:, :, i, :, :].flatten()

        ax = axes[i]
        ax.hist(flattened, bins=n_bins, log=True)
        try:
            ax.set_title(f'{metadata[0][0].bands[i]} bands (Log Scale)')
        except TypeError:
            ax.set_title(f'{metadata[0].bands[i]} bands (Log Scale)')

    plt.tight_layout()

    # fill in the code here
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "landsat_histogram.png")
        plt.close()



def plot_gt_counts(ground_truth: np.ndarray,
                   image_dir: None | str | os.PathLike = None
                   ) -> None:
    """
    This function plots the ground truth histogram over all ground truth
    values over all tiles present in the groundTruth_stack.

    Parameters
    ----------
    groundTruth : np.ndarray
        The ground truth image stack volume.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    all_gt_values = ground_truth.flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(all_gt_values, alpha=0.7, color='blue', log=True)
    plt.title('GT Histogram')

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth_histogram.png")
        plt.close()



def plot_viirs(
        viirs: np.ndarray, plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """ This function plots the VIIRS image.

    Parameters
    ----------
    viirs : np.ndarray
        The VIIRS image.
    plot_title : str
        The title of the plot.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """

    plt.figure(figsize=(8, 6))
    plt.imshow(viirs)
    plt.title(plot_title)

    # fill in the code here
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_max_projection.png")
        plt.close()


def plot_viirs_by_date(
        viirs_stack: np.array,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None) -> None:
    """
    This function plots the VIIRS image by band in subplots.

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the VIIRS image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    if viirs_stack.ndim == 5:
        vtimes = viirs_stack.shape[1]
    else:
        vtimes = viirs_stack.shape[0]

    fig, axes = plt.subplots(nrows=1, ncols=vtimes, figsize=(25,25))
    if vtimes == 1:
        axes = [axes]

    for i in range(vtimes):

        ax = axes[i]
        if viirs_stack.ndim == 5:
            timed = viirs_stack[0, i, 0, :, :]
        else:
            timed = viirs_stack[i, 0, :, :]
        ax.imshow(timed)

        try:
            ax.set_title(f"Time: {metadata[0][i].time}")
        except TypeError:
            ax.set_title(f"{metadata[i].time}")

        plt.tight_layout()

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_plot_by_date.png")
        plt.close()


def preprocess_data(
        satellite_stack: np.ndarray,
        satellite_type: str
        ) -> np.ndarray:
    """
    This function preprocesses the satellite data based on the satellite type.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    np.ndarray
    """
    if satellite_type == 'sentinel1':
        return preprocess_sentinel1(satellite_stack)
    elif satellite_type == 'sentinel2':
        return preprocess_sentinel2(satellite_stack)
    elif satellite_type == 'landsat':
        return preprocess_landsat(satellite_stack)
    elif satellite_type == 'viirs':
        return preprocess_viirs(satellite_stack)
    else:
        print("Invalid satellite type! return unchanged value")
        return satellite_stack



def create_rgb_composite_s1(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function creates an RGB composite for Sentinel-1.
    This function needs to extract the band identifiers from the metadata
    and then create the RGB composite. For the VV-VH composite, after
    the subtraction, you need to minmax scale the image.

    Parameters
    ----------
    processed_stack : np.ndarray
        The Sentinel-1 image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot. Cannot accept more than 3 bands.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    if len(bands_to_plot) > 3:
        raise ValueError("Cannot plot more than 3 bands.")

    fig, axes = plt.subplots(nrows=len(metadata)//2, ncols=2, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(len(metadata)):
        vvi = metadata[i].bands.index('VV')
        vhi = metadata[i].bands.index('VH')

        vvimg = processed_stack[i, vvi, :, :]
        vhimg = processed_stack[i, vhi, :, :]

        vv_vh = vvimg - vhimg
        mins = np.min(vv_vh)
        maxs = np.max(vv_vh)
        vv_vh_minmax = (vv_vh - mins) / (maxs - mins)
        rgb = np.stack((vvimg, vhimg, vv_vh_minmax), axis=-1)

        ax = axes[i]
        ax.imshow(rgb)
        ax.set_title(f"time: {metadata[i].time}")

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "plot_sentinel1.png")
        plt.close()



def validate_band_identifiers(
          bands_to_plot: List[List[str]],
          band_mapping: dict) -> None:
    """
    This function validates the band identifiers.

    Parameters
    ----------
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.

    Returns
    -------
    None
    """
    for value in bands_to_plot:
        for v in value:
            assert v in band_mapping, f"'{v}' not found !"


def plot_images(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        band_mapping: dict,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ):
    """
    This function plots the satellite images.

    Parameters
    ----------
    processed_stack : np.ndarray
        The satellite image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    times = processed_stack.shape[0]
    sets = len(bands_to_plot)

    fig, axes = plt.subplots(nrows=sets, ncols=times, figsize=(15, 15))

    for i in range(sets):
        for j in range(times):
            ax = axes[i, j]

            rgb = []
            for v in bands_to_plot[i]:
                idx = band_mapping[v]
                band_img = processed_stack[j, idx, :, :]
                rgb.append(band_img)

            rgb_img = np.dstack(rgb)
            ax.imshow(rgb_img)

            ax.set_title(f"Time: {metadata[j].time} Band: {bands_to_plot[i]}")

    plt.tight_layout()


    if image_dir is None:
        plt.show()
    else:
        plt.savefig(
            Path(image_dir) / f"plot_{metadata[0].satellite_type}.png"
            )
        plt.close()



def plot_satellite_by_bands(
        satellite_stack: np.ndarray,
        metadata: List[Metadata],
        bands_to_plot: List[List[str]],
        satellite_type: str,
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the satellite image by band in subplots.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    bands_to_plot : List[List[str]]
        The bands to plot.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    None
    """
    processed_stack = preprocess_data(satellite_stack, satellite_type)

    if satellite_type == "sentinel1":
        create_rgb_composite_s1(processed_stack, bands_to_plot, metadata, image_dir=image_dir)
    else:
        band_ids_per_timestamp = extract_band_ids(metadata)
        all_band_ids = [band_id for timestamp in band_ids_per_timestamp for
                        band_id in timestamp]
        unique_band_ids = sorted(list(set(all_band_ids)))
        band_mapping = {band_id: idx for
                        idx, band_id in enumerate(unique_band_ids)}
        validate_band_identifiers(bands_to_plot, band_mapping)
        plot_images(
            processed_stack,
            bands_to_plot,
            band_mapping,
            metadata,
            image_dir
            )


def extract_band_ids(metadata: List[Metadata]) -> List[List[str]]:
    """
    Extract the band identifiers from file names for each timestamp based on
    satellite type.

    Parameters
    ----------
    file_names : List[List[str]]
        A list of file names.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    List[List[str]]
        A list of band identifiers.
    """
    bilist = []

    for v in metadata:
        bilist.append(v.bands)

    return bilist



def plot_ground_truth(
        ground_truth: np.array,
        plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the groundTruth image.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.
    """

    gt = ground_truth[0, 0]

    plt.figure(figsize=(8, 6))
    plt.imshow(gt)
    plt.title(plot_title)

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()
