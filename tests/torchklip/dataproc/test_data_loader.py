import os
import pytest
import numpy as np
import tempfile
import astropy.io.fits as fits
from unittest.mock import patch, MagicMock

from torchklip.dataproc.data_loader import load_data, load_fits, FitsData

# Fixture for creating a temporary NPY file


@pytest.fixture
def temp_npy_file():
    """Create a temporary NPY file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        # Create a test array and save it to the temp file
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        np.save(tmp.name, test_array)
        tmp_path = tmp.name

    yield tmp_path

    # Clean up temp file after the test
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

# Fixture for creating a temporary NPZ file


@pytest.fixture
def temp_npz_file():
    """Create a temporary NPZ file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        # Create test arrays and save them to the temp file
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([5, 6, 7])
        np.savez(tmp.name, array1=arr1, array2=arr2)
        tmp_path = tmp.name

    yield tmp_path

    # Clean up temp file after the test
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

# Fixture for creating a temporary FITS file


@pytest.fixture
def temp_fits_file():
    """Create a temporary FITS file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        # Create a test array and header
        primary_hdu = fits.PrimaryHDU()
        test_array = np.array([[10, 20], [30, 40]])
        image_hdu = fits.ImageHDU(data=test_array)
        image_hdu.header['TESTKEY'] = 'test_value'

        # Create a FITS HDU list and write to file
        hdul = fits.HDUList([primary_hdu, image_hdu])
        hdul.writeto(tmp.name, overwrite=True)
        hdul.close()
        tmp_path = tmp.name

    yield tmp_path

    # Clean up temp file after the test
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

# Tests for load_data function


def test_load_data_npy(temp_npy_file):
    """Test loading .npy files."""
    data = load_data(temp_npy_file)

    # Verify data was loaded correctly
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3)
    assert np.array_equal(data, np.array([[1, 2, 3], [4, 5, 6]]))


def test_load_data_npz(temp_npz_file):
    """Test loading .npz files."""
    data = load_data(temp_npz_file)

    # Verify data structure and contents
    assert isinstance(data, dict)
    assert len(data) == 2
    assert 'array1' in data
    assert 'array2' in data
    assert np.array_equal(data['array1'], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(data['array2'], np.array([5, 6, 7]))


def test_load_data_fits_no_header(temp_fits_file):
    """Test loading .fits file without header."""
    data = load_data(temp_fits_file, hdu_index=1)

    # Verify data was loaded correctly
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 2)
    assert np.array_equal(data, np.array([[10, 20], [30, 40]]))


def test_load_data_fits_with_header(temp_fits_file):
    """Test loading .fits file with header."""
    data = load_data(temp_fits_file, include_header=True, hdu_index=1)

    # Verify data and header structure
    assert isinstance(data, FitsData)
    assert np.array_equal(data.data, np.array([[10, 20], [30, 40]]))
    assert 'TESTKEY' in data.header
    assert data.header['TESTKEY'] == 'test_value'


def test_load_data_fits_all_hdus(temp_fits_file):
    """Test loading all HDUs from a FITS file."""
    data_list = load_data(temp_fits_file, all_hdus=True)

    # Verify list structure and data
    assert isinstance(data_list, list)
    assert len(data_list) == 2
    # Primary HDU data may be None
    assert data_list[1].shape == (2, 2)
    assert np.array_equal(data_list[1], np.array([[10, 20], [30, 40]]))


def test_load_data_fits_all_hdus_with_header(temp_fits_file):
    """Test loading all HDUs with headers from a FITS file."""
    data_list = load_data(temp_fits_file, include_header=True, all_hdus=True)

    # Verify FitsData objects in list
    assert isinstance(data_list, list)
    assert len(data_list) == 2
    assert isinstance(data_list[1], FitsData)
    assert np.array_equal(data_list[1].data, np.array([[10, 20], [30, 40]]))
    assert 'TESTKEY' in data_list[1].header
    assert data_list[1].header['TESTKEY'] == 'test_value'


def test_load_data_unsupported_format():
    """Test error handling for unsupported file formats."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data("test_file.txt")


def test_load_data_file_not_found():
    """Test error handling for file not found."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.npy")

# Tests for load_fits function


def test_load_fits_specific_hdu(temp_fits_file):
    """Test loading a specific HDU from a FITS file."""
    data = load_fits(temp_fits_file, hdu_index=1)

    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 2)
    assert np.array_equal(data, np.array([[10, 20], [30, 40]]))


def test_load_fits_invalid_hdu_index(temp_fits_file):
    """Test error handling for invalid HDU index."""
    try:
        load_fits(temp_fits_file, hdu_index=10)
        pytest.fail("IndexError was not raised")
    except IndexError as e:
        assert "hdu_index 10 is out of range" in str(e)


def test_load_fits_mock_verify_error():
    """Test error handling for FITS verification error."""
    with patch('astropy.io.fits.open') as mock_open:
        mock_open.side_effect = fits.verify.VerifyError("Verification error")
        with pytest.raises(fits.verify.VerifyError):
            load_fits("test.fits")


def test_load_fits_mock_general_exception():
    """Test error handling for general FITS exceptions."""
    with patch('astropy.io.fits.open') as mock_open:
        mock_open.side_effect = Exception("General FITS error")
        with pytest.raises(Exception):
            load_fits("test.fits")
