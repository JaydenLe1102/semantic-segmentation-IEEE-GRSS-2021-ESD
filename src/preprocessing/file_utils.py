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
    splits = filename.split("_")
    if len(splits) != 3:
        print(f"{filename} is a invalid VIIRS filename! output will be ('-1','-1')")
        return "-1", "-1"
    else:
        return splits[2][1:-4], "0"


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
    splits = filename.split("_")
    if len(splits) != 5:
        print(f"{filename} is a invalid S1 filename! output will be ('-1','-1')")
        return "-1", "-1"
    else:
        return splits[3], splits[4][:-4]


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
    splits = filename.split("_")
    if len(splits) != 3:
        print(f"{filename} is a invalid S2 filename! output will be ('-1','-1')")
        return "-1", "-1"
    else:
        return splits[1], splits[2][1:-4]


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
    splits = filename.split("_")
    if len(splits) != 4:
        print(f"{filename} is a invalid Landsat filename! output will be ('-1','-1')")
        return "-1", "-1"
    else:
        return splits[2], splits[3][1:-4]


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
    return "0", "0"


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
    patterns = {
        "viirs": "DNB_VNP46A1_*",
        "sentinel1": "S1A_IW_GRDH_*",
        "sentinel2": "L2A_*",
        "landsat": "LC08_L1TP_*",
        "gt": "groundTruth.tif",
    }

    if satellite_type not in patterns:
        print(f"{satellite_type} is a invalid satellite type, a empty list will be returned")
        return []
    else:
        result = Path(tile_dir).glob(patterns[satellite_type])
        return list(result)


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
    patterns = {
        "viirs": "DNB_VNP46A1_*",
        "sentinel1": "S1A_IW_GRDH_*",
        "sentinel2": "L2A_*",
        "landsat": "LC08_L1TP_*",
        "gt": "groundTruth.tif",
    }

    if satellite_type not in patterns:
        print(f"{satellite_type} is a invalid satellite type, '-1' will be returned")
        return "PATTERN_NOT_FOUND"
    else:
        return patterns[satellite_type]


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
    arr = []
    for v in sat_files:
        image = tifffile.imread(v)
        arr.append(image)

    return arr


def stack_satellite_data(
        sat_data: List[np.ndarray],
        file_names: List[str],
        satellite_type: str
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
    groupf = get_grouping_function(satellite_type)
    # Apply the grouping function to each file name to get the date and band
    datas = [groupf(Path(v).name) for v in file_names]
    # Sort the satellite data and file names based on the date and band
    combined_d = zip(sat_data, file_names, datas)
    sorted_d = sorted(combined_d, key=lambda x: x[2])
    # Initialize lists to store the stacked data and metadata
    stacked_d = []
    metadata = []
    # Group the data by date
    for v1, v2 in groupby(sorted_d, lambda x: x[2][0]):
        # Sort the group by band
        sorted_b = sorted(v2, key=lambda x:x[2][1])
        # Extract the date and band, satellite data, and file names from the
        # sorted group
        sdata = [item[0] for item in sorted_b]
        filename = [item[1] for item in sorted_b]
        bands = [item[2][1] for item in sorted_b]  # date should be v1
        # Stack the satellite data along a new axis and append it to the list
        np_stacked = np.stack(sdata)
        stacked_d.append(np_stacked)
        # Create a Metadata object and append it to the list
        new_meta = Metadata(satellite_type, filename, "Need a str here", bands,
                            v1)
        metadata.append(new_meta)
    # Stack the list of satellite data arrays along a new axis to create a
    # 4D array with dimensions (time, band, height, width)
    result = np.stack(stacked_d)

    # Return the stacked satellite data and the list Metadata objects.
    return result, metadata


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
    if satellite_type == 'viirs':
        return process_viirs_filename
    elif satellite_type == 'sentinel1':
        return process_s1_filename
    elif satellite_type == 'sentinel2':
        return process_s2_filename
    elif satellite_type == 'landsat':
        return process_landsat_filename
    elif satellite_type == 'gt':
        return process_ground_truth_filename
    else:
        print(f"{satellite_type} is invalid! None would be returned")
        return None


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

    dates = set()
    bands = set()

    for v1, v2 in metadata_keys:
        dates.add(v1)
        bands.add(v2)

    return dates, bands


def load_satellite(
        tile_dir: str | os.PathLike,
        satellite_type: str
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

    paths = Path(tile_dir)
    sfile = get_satellite_files(paths, satellite_type)
    npdata = read_satellite_files(sfile)
    stacked_d, metadata = stack_satellite_data(npdata, [v.name for v in sfile], satellite_type)

    return stacked_d, metadata


def load_satellite_dir(
        data_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[List[Metadata]]]:
    """
    Load all bands for a given satellite type from a directory of multiple
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
    paths = Path(data_dir)
    sfile = [v for v in paths.glob("*") if v.is_dir()]

    all_sd = []
    all_md = []

    for v in sfile:
        stacked_d, metadata = load_satellite(v, satellite_type)
        all_sd.append(stacked_d)
        all_md.append(metadata)

    result = np.stack(all_sd)

    return result, all_md

