# torchklip/dataproc/data_loader.py
import os
from dataclasses import dataclass
from typing import Any, Union, List

import numpy as np
import astropy.io.fits as fits

# local modules
from ..utils.logging_utils import get_logger

# Get a logger specific to this module
logger = get_logger(__name__.split('.')[-1])


@dataclass
class FitsData:
    """
    Container for FITS file data and header information.

    Attributes:
        data (Any): The primary data, typically a NumPy array.
        header (Any): The header from the FITS file (astropy.io.fits.Header).
    """
    data: Any
    header: Any


def load_data(file_path: Union[str, os.PathLike], include_header: bool = False, **kwargs) -> Any:
    """
    Loads data from .npz, .npy, or .fits files.

    For FITS files, additional keyword arguments (e.g., `hdu_index` or `all_hdus`) 
    are passed to `load_fits`.

    Args:
        file_path (str or os.PathLike): Path to the file.
        include_header (bool): If True (for FITS files), returns data with the header.
        **kwargs: Additional options for FITS file loading.

    Returns:
        np.ndarray or FitsData or List[Union[np.ndarray, FitsData]]:
            - `.npz` or `.npy`: Returns the NumPy array.
            - `.fits`: Returns data, a `FitsData` instance, or a list of them.

    Raises:
        ValueError: If the file format is unsupported.
    """
    file_path = os.fspath(file_path)

    # Handle NumPy compressed archives (.npz)
    if file_path.endswith('.npz'):
        try:
            # Load the .npz file with pickle support enabled
            with np.load(file_path, allow_pickle=True) as npz_file:

                # Special case: single array saved without a key
                if 'arr_0' in npz_file:
                    arr = npz_file['arr_0']
                    data = arr.copy() if hasattr(arr, 'copy') else arr

                # Multiple arrays saved with keys
                else:
                    data = {}
                    for k in npz_file.keys():
                        if k != 'arr_0':
                            arr = npz_file[k]
                            data[k] = arr.copy() if hasattr(
                                arr, 'copy') else arr
                # Log successful load and contents information
                logger.info(f"Successfully loaded .npz file: {file_path}")
                if hasattr(data, 'keys'):
                    logger.info(f"Contains keys: {list(data.keys())}")

                return data

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise

        except Exception as e:
            logger.error(f"Error loading .npz file: {str(e)}")
            raise

    # Handle NumPy binary files (.npy)
    elif file_path.endswith('.npy'):
        try:
            data = np.load(file_path)
            logger.info(f"Successfully loaded .npy file: {file_path}")
            logger.info(
                f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

            return data

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise

        except Exception as e:
            logger.error(f"Error loading .npy file: {str(e)}")
            raise

    # Handle FITS astronomical data files
    elif file_path.endswith('.fits'):
        return load_fits(file_path, include_header=include_header, **kwargs)

    # Reject unsupported file formats
    else:
        supported_formats = ['.npz', '.npy', '.fits']
        error_msg = f"Unsupported file format: {file_path}. Only {', '.join(supported_formats)} files are supported."
        logger.error(error_msg)
        raise ValueError(error_msg)


def load_fits(file_path: str, include_header: bool = False, hdu_index: int = 0,
              all_hdus: bool = False) -> Union[np.ndarray, FitsData, List[Union[np.ndarray, FitsData]]]:
    """
    Loads data from a FITS file.

    This function supports:
    1. Loading a specific HDU (`hdu_index`, default: 0).
    2. Loading all HDUs if `all_hdus` is True.

    Args:
        file_path (str): Path to the FITS file.
        include_header (bool): If True, includes header information.
        hdu_index (int): Index of the HDU to load (default: 0). Ignored if `all_hdus` is True.
        all_hdus (bool): If True, loads all HDUs instead of a single one.

    Returns:
        np.ndarray, FitsData, or List[Union[np.ndarray, FitsData]]:
            - If `all_hdus` is False:
                - Returns `FitsData` if `include_header` is True.
                - Otherwise, returns just the data array.
            - If `all_hdus` is True:
                - Returns a list of `FitsData` instances (if `include_header` is True).
                - Otherwise, returns a list of data arrays.

    Raises:
        IndexError: If `hdu_index` is out of range.
    """
    try:
        with fits.open(file_path) as hdul:

            # Load all HDUs if requested
            if all_hdus:
                data_list = [FitsData(hdu.data, hdu.header)
                             if include_header else hdu.data for hdu in hdul]
                logger.info(f"Successfully loaded .fits file: {file_path}")
                logger.info(f"Loaded {len(hdul)} HDU(s)")
                return data_list

            # Check if the specified hdu_index is within the valid range
            if not (0 <= hdu_index < len(hdul)):
                raise IndexError(
                    f"hdu_index {hdu_index} is out of range. The file contains {len(hdul)} HDU(s).")

            # Load the specified HDU
            selected_hdu = hdul[hdu_index]
            logger.info(f"Successfully loaded .fits file: {file_path}")
            logger.info(f"HDU {hdu_index}: shape {selected_hdu.data.shape}")
            data = FitsData(
                selected_hdu.data, selected_hdu.header) if include_header else selected_hdu.data
            return data

    # Handle file not found errors explicitly
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    # Handle FITS verification errors
    except fits.verify.VerifyError as e:
        logger.error(f"FITS verification error: {str(e)}")
        raise

    # Handle general exceptions from FITS
    except Exception as e:
        logger.error(f"Error processing FITS file: {str(e)}")
        raise


__all__ = ["load_data"]
