""" This code applies preprocessing functions on the IEEE GRSS ESD satellite data."""
import numpy as np
from scipy.ndimage import gaussian_filter


def per_band_gaussian_filter(img: np.ndarray, sigma: float = 1):
    """
    For each band in the image, apply a gaussian filter with the given sigma.
    The gaussian filter should be applied to each heightxwidth image individually.

    Parameters
    ----------
    img : np.ndarray
        The image to be filtered. The shape of the array is (time, band, height, width).
    sigma : float
        The sigma of the gaussian filter.

    Returns
    -------
    np.ndarray
        The filtered image. The shape of the array is (time, band, height, width).
    """
    gaussed_img = np.zeros_like(img)

    for v1 in range(img.shape[0]):
        for v2 in range(img.shape[1]):
            gaussed_img[v1, v2] = gaussian_filter(img[v1, v2], sigma=sigma)

    return gaussed_img


def quantile_clip(img_stack: np.ndarray,
                  clip_quantile: float,
                  group_by_time=True
                  ) -> np.ndarray:
    """
    This function clips the outliers of the image stack by the given quantile.
    It calculates the top `clip_quantile` samples and the bottom `clip_quantile`
    samples, and sets any value above the top to the first value under the top value,
    and any value below the bottom to the first value above the top value.

    group_by_time affects how img_max and img_min are calculated, if
    group_by_time is true, the quantile limits are shared along the time dimension.
    Otherwise, the quantile limits are calculated individually for each image.

    Parameters
    ----------
    img_stack : np.ndarray
        The image stack to be clipped. The shape of the array is (time, band, height, width).
    clip_quantile : float
        The quantile to clip the outliers by. Value between 0 and 0.5.

    Returns
    -------
    np.ndarray
        The clipped image stack. The shape of the array is (time, band, height, width).
    """
    if clip_quantile < 0 or clip_quantile > 0.5:
        print("quantile should between 0 and 0.5, so a unchanged array would be returned")
        return img_stack

    if group_by_time:
        axes = (0, 2, 3)
    else:
        axes = (2, 3)

    lower = np.quantile(img_stack, clip_quantile, axis=axes, keepdims=True)
    upper = np.quantile(img_stack, 1 - clip_quantile, axis=axes, keepdims=True)

    img_stack = np.clip(img_stack, lower, upper)
    return img_stack


def minmax_scale(img: np.ndarray, group_by_time=True):
    """
    This function minmax scales the image stack to values between 0 and 1.
    This transforms any image to have a range between img_min to img_max
    to an image with the range 0 to 1, using the formula 
    (pixel_value - img_min)/(img_max - img_min).

    group_by_time affects how img_max and img_min are calculated, if
    group_by_time is true, the min and max are shared along the time dimension.
    Otherwise, the min and max are calculated individually for each image.
    
    Parameters
    ----------
    img : np.ndarray
        The image stack to be minmax scaled. The shape of the array is (time, band, height, width).
    group_by_time : bool
        Whether to group by time or not.

    Returns
    -------
    np.ndarray
        The minmax scaled image stack. The shape of the array is (time, band, height, width).
    """

    if group_by_time:
        axes = (0, 2, 3)
    else:
        axes = (2, 3)

    mins = np.min(img, axis=axes, keepdims=True)
    maxs = np.max(img, axis=axes, keepdims=True)

    result = (img - mins) / (maxs - mins)

    return result


def brighten(img, alpha=0.13, beta=0):
    """
    Function to brighten the image.
    This is calculated using the formula new_pixel = alpha*pixel+beta.
    If a value of new_pixel falls outside of the [0,1) range,
    the values are clipped to be 0 if the value is under 0 and 1 if the value is over 1.

    Parameters
    ----------
    img : np.ndarray
        The image to be brightened. The shape of the array is (time, band, height, width).
        The input values are between 0 and 1.
    alpha : float
        The alpha parameter of the brightening.
    beta : float
        The beta parameter of the brightening.

    Returns
    -------
    np.ndarray
        The brightened image. The shape of the array is (time, band, height, width).
    """

    brightened = img * alpha + beta
    cliped_bri = np.clip(brightened, 0, 1)
    return cliped_bri


def gammacorr(band, gamma=2):
    """
    This function applies a gamma correction to the image.
    This is done using the formula pixel^(1/gamma)

    Parameters
    ----------
    band : np.ndarray
        The image to be gamma corrected. The shape of the array is (time, band, height, width).
        The input values are between 0 and 1.
    gamma : float
        The gamma parameter of the gamma correction.

    Returns
    -------
    np.ndarray
        The gamma corrected image. The shape of the array is (time, band, height, width).
    """
    if gamma <= 0:
        print("we can't have gamma <= 0 for 1/gamma! an unchanged band would be returned")
        return band

    gamma_ed = np.power(band, 1 / gamma)
    return gamma_ed


def maxprojection_viirs(
        viirs_stack: np.ndarray,
        clip_quantile: float = 0.01
        ) -> np.ndarray:
    """
    This function takes a stack of VIIRS tiles and returns a single
    image that is the max projection of the tiles.

    The output value of the projection is such that 
    output[band,i,j] = max_time(input[time,band,i,j])
    i.e, the value of a pixel is the maximum value over all time steps.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles. The shape of the array is (time, band, height, width).

    Returns
    -------
    np.ndarray
        Max projection of the VIIRS stack, of shape (band, height, width)
    """
    # HINT: use the time dimension to perform the max projection over.
    if clip_quantile < 0 or clip_quantile > 0.5:
        print("quantile should between 0 and 0.5, so a unchanged array would be returned")
        return viirs_stack

    lower = np.quantile(viirs_stack, clip_quantile, axis=(1, 2, 3), keepdims=True)
    upper = np.quantile(viirs_stack, 1 - clip_quantile, axis=(1, 2, 3), keepdims=True)
    vstack = np.clip(viirs_stack, lower, upper)

    maxp_stack = np.max(vstack, axis=0)

    mins = np.min(maxp_stack)
    maxs = np.max(maxp_stack)
    result = (maxp_stack - mins) / (maxs - mins)

    return result


def preprocess_sentinel1(
        sentinel1_stack: np.ndarray,
        clip_quantile: float = 0.01,
        sigma=1
        ) -> np.ndarray:
    """
    In this function we will preprocess sentinel1. The steps for preprocessing
    are the following:
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gaussian filter
        - Minmax scale
    """
    db_s1 = 10 * np.log10(sentinel1_stack)

    clipped_s1 = quantile_clip(db_s1, clip_quantile)

    gaussed_s1 = per_band_gaussian_filter(clipped_s1, sigma)

    minmax_s1 = minmax_scale(gaussed_s1, group_by_time=True)

    return minmax_s1


def preprocess_sentinel2(sentinel2_stack: np.ndarray,
                         clip_quantile: float = 0.05,
                         gamma: float = 2.2
                         ) -> np.ndarray:
    """
    In this function we will preprocess sentinel-2. The steps for
    preprocessing are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    clipped_s2 = quantile_clip(sentinel2_stack, clip_quantile)

    gamma_s2 = gammacorr(clipped_s2, gamma)

    minmax_s2 = minmax_scale(gamma_s2, group_by_time=True)

    return minmax_s2


def preprocess_landsat(
        landsat_stack: np.ndarray,
        clip_quantile: float = 0.05,
        gamma: float = 2.2
        ) -> np.ndarray:
    """
    In this function we will preprocess landsat. The steps for preprocessing
    are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    clipped_ls = quantile_clip(landsat_stack, clip_quantile)

    gamma_ls = gammacorr(clipped_ls, gamma)

    minmax_ls = minmax_scale(gamma_ls, group_by_time=True)

    return minmax_ls


def preprocess_viirs(viirs_stack, clip_quantile=0.05) -> np.ndarray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Minmax scale
    """
    clipped_vs = quantile_clip(viirs_stack, clip_quantile)

    minmax_vs = minmax_scale(clipped_vs, group_by_time=True)

    return minmax_vs
