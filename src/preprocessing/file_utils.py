"""
This module contains functions for loading satellite data from a directory of
tiles.
"""

from pathlib import Path
from typing import Tuple, List, Set
import os
from itertools import groupby
import re
from dataclasses import dataclass
import tifffile
import numpy as np


@dataclass
class Metadata:
    """
    A class to store metadata about a stack of satellite files from the same date.

    The attributes are the following:
    satellite_type: one of "viirs", "sentinel1", "sentinel2", "landsat", or "gt"
    file_name: a list of the original filenames of the satellite's bands
    tile_id: name of the tile directory, i.e., "Tile1", "Tile2", etc
    bands: a list of the names of the bands with correspondence to the
    indexes of the stack object, i.e. ["VH", "VV"] for sentinel-1
    time: time of the observations
    """

    satellite_type: str
    file_name: List[str]
    tile_id: str
    bands: List[str]
    time: str


def process_viirs_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a VIIRS file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: DNB_VNP46A1_A2020221.tif
    Example output: ("2020221", "0")

    Parameters
    ----------
    filename : str
        The filename of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    date = filename.split("_")[2].split(".")[0]
    date = date[1:]

    band = "0"

    return (date, band)


def process_s1_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-1 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: S1A_IW_GRDH_20200804_VV.tif
    Example output: ("20200804", "VV")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    date = filename.split("_")[3]

    band = filename.split("_")[4].split(".")[0]

    return (date, band)


def process_s2_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-2 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: L2A_20200816_B01.tif
    Example output: ("20200804", "01")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-2 file.

    Returns
    -------
    Tuple[str, str]
    """
    date = filename.split("_")[1]
    band = filename.split("_")[2].split(".")[0]
    band = band[1:]
    return (date, band)


def process_landsat_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band)

    Example input: LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "9")

    Parameters
    ----------
    filename : str
        The filename of the Landsat file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """

    date = filename.split("_")[2]
    band = filename.split("_")[3].split(".")[0]
    band = band[1:]
    return (date, band)


def process_ground_truth_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file.

    Example input: groundTruth.tif
    Example output: ("0", "0")

    Parameters
    ----------
    filename: str
        The filename of the ground truth file though we will ignore it.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    date = "0"
    band = "0"
    return (date, band)


def get_satellite_files(tile_dir: Path, satellite_type: str) -> List[Path]:
    """
    Retrieve all satellite files matching the satellite type pattern.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file.
    """

    regex_patterns = {
        "viirs": r"^DNB_VNP46A1_.*\.tif$",
        "sentinel1": r"^S1A_IW_GRDH_.*\.tif$",
        "sentinel2": r"^L2A_.*\.tif$",
        "landsat": r"^LC08_L1TP_.*\.tif$",
        "gt": r"^groundTruth\.tif$",
    }

    matching_list = []
    regex_patterns = re.compile(get_filename_pattern(satellite_type))

    for i in tile_dir.iterdir():
        if regex_patterns.match(i.name):
            matching_list.append(i)

    return matching_list


def get_filename_pattern(satellite_type: str) -> str:
    """
    Return the filename pattern for the given satellite type.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    str
        The filename pattern for the given satellite type.
    """
    if satellite_type == "sentinel1":
        return "S1A_IW_GRDH_*"
    elif satellite_type == "sentinel2":
        return "L2A_*"
    elif satellite_type == "landsat":
        return "LC08_L1TP_*"
    elif satellite_type == "viirs":
        return "DNB_VNP46A1_*"
    elif satellite_type == "gt":
        return "groundTruth.tif"


def read_satellite_files(sat_files: List[Path]) -> List[np.ndarray]:
    """
    Read satellite files into a list of numpy arrays.

    Parameters
    ----------
    sat_files : List[Path]
        A list of Path objects for each satellite file.

    Returns
    -------
    List[np.ndarray]

    """
    satellite_data = []

    for i in sat_files:
        data = tifffile.imread(i)
        satellite_data.append(data)

    return satellite_data


def stack_satellite_data(
    sat_data: List[np.ndarray], file_names: List[str], satellite_type: str
) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Stack satellite data into a single array and collect filenames.

    Parameters
    ----------
    sat_data : List[np.ndarray]
        A list containing the image data for all bands with respect to
        a single satellite (sentinel-1, sentinel-2, landsat-8, or viirs)
        at a specific timestamp.
    file_names : List[str]
        A list of filenames corresponding to the satellite and timestamp.

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with dimensions
        (time, band, height, width) and a list of the filenames.
    """
    # Get the function to group the data based on the satellite type
    group_func = get_grouping_function(satellite_type)
    # Apply the grouping function to each file name to get the date and band
    dates = []
    bands = []

    tile_id = 1
    for i in file_names:
        if isinstance(i, Path):
            tile_id = i.parts[-2]
            i = i.name

        date, band = group_func(i)
        dates.append(date)
        bands.append(band)

    # Sort the satellite data and file names based on the date and band
    sorted_order = np.lexsort((bands, dates))

    sorted_sat_data = []
    sorted_file_names = []
    sorted_info = []
    for i in sorted_order:
        sorted_sat_data.append(sat_data[i])
        sorted_file_names.append(file_names[i])
        sorted_info.append((dates[i], bands[i]))

    # Initialize lists to store the stacked data and metadata
    data_list = []
    metadata_list = []

    # Group the data by date
    grouped_by_date = groupby(
        list(zip(sorted_info, sorted_sat_data, sorted_file_names)),
        key=lambda x: x[0][0],
    )
    # Sort the group by band

    for date, group in grouped_by_date:
        sorted_by_band = sorted(group, key=lambda x: x[0][1])

        list_band = []
        list_file_names = []
        list_data = []

        for (date, band), data, filename in sorted_by_band:
            list_band.append(band)
            list_file_names.append(filename)
            list_data.append(data)
        data_list.append(np.stack(list_data, axis=0))
        metadata = Metadata(
            satellite_type=satellite_type,
            file_name=list_file_names,
            tile_id=tile_id,
            bands=list_band,
            time=date,
        )
        metadata_list.append(metadata)

    return np.stack(data_list, axis=0), metadata_list


def get_grouping_function(satellite_type: str):
    """
    Return the function to group satellite files by date and band.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    function
        The function to group satellite files by date and band.
    """
    if satellite_type == "viirs":
        return process_viirs_filename
    elif satellite_type == "sentinel1":
        return process_s1_filename
    elif satellite_type == "sentinel2":
        return process_s2_filename
    elif satellite_type == "landsat":
        return process_landsat_filename
    elif satellite_type == "gt":
        return process_ground_truth_filename


def get_unique_dates_and_bands(
    metadata_keys: Set[Tuple[str, str]]
) -> Tuple[Set[str], Set[str]]:
    """
    Extract unique dates and bands from satellite metadata keys.

    Parameters
    ----------
    metadata_keys : Set[Tuple[str, str]]
        A set of tuples containing the date and band for each satellite file.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        A tuple containing the unique dates and bands.
    """

    unique_dates = set()
    unique_bands = set()

    for date, band in metadata_keys:
        unique_dates.add(date)
        unique_bands.add(band)

    return unique_dates, unique_bands


def load_satellite(
    tile_dir: str | os.PathLike, satellite_type: str
) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Load all bands for a given satellite type from a directory of tile files.

    Parameters
    ----------
    tile_dir : str or os.PathLike
        The Tile directory containing the satellite tiff files.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with
        dimensions (time, band, height, width) and a list of the filenames.
    """

    if isinstance(tile_dir, str):
        tile_dir = Path(tile_dir)

    list_files = get_satellite_files(tile_dir, satellite_type)

    list_datas = read_satellite_files(list_files)

    return stack_satellite_data(list_datas, list_files, satellite_type)


def load_satellite_dir(
    data_dir: str | os.PathLike, satellite_type: str
) -> Tuple[np.ndarray, List[List[Metadata]]]:
    """
        Load all bands for a given satellite type fhttps://drive.google.com/file/d/FILE_ID/view?usp=sharing
    rom a directory of multiple
        tile files.

        Parameters
        ----------
        data_dir : str or os.PathLike
            The directory containing the satellite tiles.
        satellite_type : str
            The type of satellite, one of "viirs", "sentinel1", "sentinel2",
            "landsat", "gt"

        Returns
        -------
        Tuple[np.ndarray, List[List[Metadata]]]
            A tuple containing the satellite data as a volume with
            dimensions (tile_dir, time, band, height, width) and a list of the
            Metadata objects.
    """

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    list_tile_datas = []
    list_tile_metadatas = []
    for tile_dir in data_dir.iterdir():
        if not tile_dir.is_dir():
            raise ValueError(f"{tile_dir} is not a directory")

        tile_datas, tile_metadatas = load_satellite(tile_dir, satellite_type)
        list_tile_datas.append(tile_datas)
        list_tile_metadatas.append(tile_metadatas)

    return np.stack(list_tile_datas, axis=0), list_tile_metadatas
