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
    preprocess_viirs,
)


def plot_viirs_histogram(
    viirs_stack: np.ndarray, image_dir: None | str | os.PathLike = None, n_bins=100
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
    # fill in the code here

    viirs_values = viirs_stack.flatten()
    plt.hist(viirs_values, bins=n_bins, log=True)
    plt.title("VIIRS Histogram In Log Scale")
    plt.xlabel("Pixel Values")
    plt.ylabel("Frequency")
    #

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "VIIRS_histogram.png")
        plt.close()


def plot_sentinel1_histogram(
    sentinel1_stack: np.ndarray,
    metadata: List[Metadata],
    image_dir: None | str | os.PathLike = None,
    n_bins=20,
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
    # fill in the code here

    num_bands = sentinel1_stack.shape[2]
    fig, axes = plt.subplots(1, num_bands, figsize=(15, 5))

    for i in range(num_bands):
        sentinel1_values = sentinel1_stack[:, :, i, :, :].flatten()
        axes[i].hist(sentinel1_values, bins=n_bins, log=True)
        axes[i].set_title(f"Band {metadata[0][0].bands[i]} Histogram In Log Scale")
        axes[i].set_xlabel("Pixel Values")
        axes[i].set_ylabel("Frequency")
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel1_histogram.png")
        plt.close()


def plot_sentinel2_histogram(
    sentinel2_stack: np.ndarray,
    metadata: List[List[Metadata]],
    image_dir: None | str | os.PathLike = None,
    n_bins=20,
) -> None:
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
    # fill in the code here

    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    col_index = 0
    row_index = 0
    num_bands = sentinel2_stack.shape[2]

    for i in range(num_bands):
        sentinel2_values = sentinel2_stack[:, :, i, :, :].flatten()

        axes[row_index, col_index].hist(sentinel2_values, bins=n_bins, log=True)
        axes[row_index, col_index].set_title(
            f"Band {metadata[0][0].bands[i]} Histogram In Log Scale"
        )
        axes[row_index, col_index].set_xlabel("Pixel Values")
        axes[row_index, col_index].set_ylabel("Frequency")

        if col_index == 2:
            col_index = 0
            row_index += 1
        else:
            col_index += 1

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
    n_bins=20,
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
    # fill in the code here

    fig, axes = plt.subplots(4, 3, figsize=(15, 15))

    col_index = 0
    row_index = 0
    num_bands = landsat_stack.shape[2]

    for i in range(num_bands):
        landsat_values = landsat_stack[:, :, i, :, :].flatten()

        axes[row_index, col_index].hist(landsat_values, bins=n_bins, log=True)
        axes[row_index, col_index].set_title(
            f"Band {metadata[0][0].bands[i]} Histogram In Log Scale"
        )
        axes[row_index, col_index].set_xlabel("Pixel Values")
        axes[row_index, col_index].set_ylabel("Frequency")

        if col_index == 2:
            col_index = 0
            row_index += 1
        else:
            col_index += 1

    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "landsat_histogram.png")
        plt.close()


def plot_gt_counts(
    ground_truth: np.ndarray, image_dir: None | str | os.PathLike = None
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
    # fill in the code here

    unique_classes, counts = np.unique(ground_truth, return_counts=True)

    plt.bar(unique_classes, counts)
    plt.title("Ground Truth Counts")
    plt.xlabel("Class Labels")
    plt.ylabel("Counts")

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth_histogram.png")
        plt.close()


def plot_viirs(
    viirs: np.ndarray, plot_title: str = "", image_dir: None | str | os.PathLike = None
) -> None:
    """This function plots the VIIRS image.

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

    plt.imshow(viirs)
    plt.title(plot_title)
    plt.colorbar(label="Pixel Values")

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_max_projection.png")
        plt.close()


def plot_viirs_by_date(
    viirs_stack: np.array,
    metadata: List[List[Metadata]],
    image_dir: None | str | os.PathLike = None,
) -> None:
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
    # fill in the code here

    num_dates = viirs_stack.shape[0]

    num_rows = num_dates // 3

    figure, axes = plt.subplots(num_rows, 3, figsize=(15, 10))

    col_index = 0
    row_index = 0
    for date in range(num_dates):
        viirs_image = viirs_stack[date, 0, :, :]
        axes[row_index, col_index].imshow(viirs_image)
        axes[row_index, col_index].set_title(f"Date {date+1}")
        axes[row_index, col_index].axis("off")

        if col_index == 2:
            col_index = 0
            row_index += 1
        else:
            col_index += 1

    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_plot_by_date.png")
        plt.close()


def preprocess_data(satellite_stack: np.ndarray, satellite_type: str) -> np.ndarray:
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
    if satellite_type == "sentinel1":
        return preprocess_sentinel1(satellite_stack)

    elif satellite_type == "sentinel2":
        return preprocess_sentinel2(satellite_stack)

    elif satellite_type == "landsat":
        return preprocess_landsat(satellite_stack)

    elif satellite_type == "viirs":
        return preprocess_viirs(satellite_stack)

    else:
        raise ValueError(f"Unsupported satellite type: {satellite_type}")


def create_rgb_composite_s1(
    processed_stack: np.ndarray,
    bands_to_plot: List[List[str]],
    metadata: List[List[Metadata]],
    image_dir: None | str | os.PathLike = None,
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

    # fill in the code here

    times = processed_stack.shape[0]
    figure, axes = plt.subplots(
        len(bands_to_plot), times, figsize=(times * 7, len(bands_to_plot) * 7)
    )

    for time in range(times):
        multi_img_channels = []
        for band, band_group in enumerate(bands_to_plot):
            for band_id in band_group:
                if band_id not in metadata[time].bands:
                    continue
                band_index = metadata[time].bands.index(band_id)
                multi_img_channels.append(
                    minmax_scale(np.array([[processed_stack[time, band_index]]]))
                )

        multi_img_channels.append(
            minmax_scale(multi_img_channels[0] - multi_img_channels[1])
        )
        img = np.stack(multi_img_channels, axis=-1)
        axes[time].imshow(img[0][0])
        axes[time].set_title(f"Plot Time: {metadata[time].time}")
        axes[time].axis("off")

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "plot_sentinel1.png")
        plt.close()


def validate_band_identifiers(
    bands_to_plot: List[List[str]], band_mapping: dict
) -> None:
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

    for band_group in bands_to_plot:
        for band_id in band_group:
            if band_id not in band_mapping:
                raise ValueError(f"Invalid band identifier: {band_id}")


def plot_images(
    processed_stack: np.ndarray,
    bands_to_plot: List[List[str]],
    band_mapping: dict,
    metadata: List[List[Metadata]],
    image_dir: None | str | os.PathLike = None,
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
    # fill in the code here

    time = processed_stack.shape[0]

    figure, axes = plt.subplots(
        len(bands_to_plot[0]), time, figsize=(time * 7, len(bands_to_plot[0]) * 7)
    )

    column_index = 0
    row_index = 0

    for time_index in range(time):
        for band, band_group in enumerate(bands_to_plot):
            image_stack_channels = []
            for band_id in band_group:

                band_index = band_mapping[band_id]

                image_stack_channels.append(processed_stack[time_index, band_index])

            img = np.stack(image_stack_channels, axis=-1)
            axes[row_index, column_index].imshow(img)

            axes[row_index, column_index].set_title(
                f"Plot Time: {metadata[time_index].time} with {band_group}"
            )
            axes[row_index, column_index].axis("off")

            if row_index < len(bands_to_plot[0]) - 1:
                row_index += 1
            else:
                row_index = 0
        column_index += 1

    #
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / f"plot_{metadata[0].satellite_type}.png")
        plt.close()


def plot_satellite_by_bands(
    satellite_stack: np.ndarray,
    metadata: List[Metadata],
    bands_to_plot: List[List[str]],
    satellite_type: str,
    image_dir: None | str | os.PathLike = None,
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
        create_rgb_composite_s1(
            processed_stack, bands_to_plot, metadata, image_dir=image_dir
        )
    else:
        band_ids_per_timestamp = extract_band_ids(metadata)
        all_band_ids = [
            band_id for timestamp in band_ids_per_timestamp for band_id in timestamp
        ]
        unique_band_ids = sorted(list(set(all_band_ids)))
        band_mapping = {band_id: idx for idx, band_id in enumerate(unique_band_ids)}
        validate_band_identifiers(bands_to_plot, band_mapping)
        plot_images(processed_stack, bands_to_plot, band_mapping, metadata, image_dir)


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

    bands = []

    for meta in metadata:
        bands.append(meta.bands)

    return bands


def plot_ground_truth(
    ground_truth: np.array,
    plot_title: str = "",
    image_dir: None | str | os.PathLike = None,
) -> None:
    """
    This function plots the groundTruth image.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.
    """
    # fill in the code here

    plt.imshow(ground_truth[0][0])
    plt.title(plot_title)
    plt.colorbar(label="Pixel Values")

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()
