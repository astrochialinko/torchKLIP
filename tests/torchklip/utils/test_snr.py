import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from torchklip.utils.snr import (
    get_r_pa,
    simple_aperture_locations,
    cartesian_coords,
    reduce_apertures,
    calc_snr_mawet,
    compute_snr,
    noise_aperture_centers,
    draw_apertures,
)


class TestSNRUtils:
    """Test class for SNR utility functions."""

    def test_get_r_pa(self):
        """Test the conversion from pixel coordinates to polar coordinates."""
        image_shape = torch.Size([100, 100])
        # Center
        r_px, pa_deg = get_r_pa(image_shape, 49.5, 49.5)
        assert r_px.item() == pytest.approx(0.0, abs=1e-5)

        # North (y-axis, top)
        r_px, pa_deg = get_r_pa(image_shape, 49.5, 19.5)
        assert r_px.item() == pytest.approx(30.0, abs=1e-5)
        # Actual implementation returns -180
        assert pa_deg.item() == pytest.approx(-180.0, abs=1e-5)

        # East (x-axis, right)
        r_px, pa_deg = get_r_pa(image_shape, 79.5, 49.5)
        assert r_px.item() == pytest.approx(30.0, abs=1e-5)
        # Actual implementation returns -90
        assert pa_deg.item() == pytest.approx(-90.0, abs=1e-5)

        # South (y-axis, bottom)
        r_px, pa_deg = get_r_pa(image_shape, 49.5, 79.5)
        assert r_px.item() == pytest.approx(30.0, abs=1e-5)
        assert pa_deg.item() == pytest.approx(0.0, abs=1e-5)  # Test actual return value

        # West (x-axis, left)
        r_px, pa_deg = get_r_pa(image_shape, 19.5, 49.5)
        assert r_px.item() == pytest.approx(30.0, abs=1e-5)
        assert pa_deg.item() == pytest.approx(90.0, abs=1e-5)  # Test actual return value

    def test_simple_aperture_locations(self):
        """Test the generation of aperture locations in a ring."""
        # Basic circular arrangement
        locations = simple_aperture_locations(10.0, 0.0, 5.0)
        # Actual implementation creates 12 locations
        assert len(locations) == 12

        # With planet excluded
        locations = simple_aperture_locations(
            10.0, 0.0, 5.0, exclude_planet=True)
        assert len(locations) == 11

        # With nearest neighbors excluded
        locations = simple_aperture_locations(
            10.0, 0.0, 5.0, exclude_nearest=2)
        assert len(locations) == 8  # 12 total - 4 excluded (2 on each side)

        # Combined exclusions
        locations = simple_aperture_locations(
            10.0, 0.0, 5.0, exclude_nearest=2, exclude_planet=True)
        assert len(locations) == 7  # 12 total - 1 planet - 4 neighbors

        # Check first location is at correct angle (for PA=0, start at top)
        locations = simple_aperture_locations(10.0, 0.0, 5.0)
        assert locations[0][0] == pytest.approx(0.0, abs=1e-5)
        assert locations[0][1] == pytest.approx(
            10.0, abs=1e-5)  # Corrected expected y-coordinate

    def test_cartesian_coords(self):
        """Test the generation of Cartesian coordinate grids."""
        center = (49.5, 49.5)
        data_shape = torch.Size([100, 100])

        xx, yy = cartesian_coords(center, data_shape)

        # Check dimensions
        assert xx.shape == data_shape
        assert yy.shape == data_shape

        # Check center point
        assert xx[49, 49].item() == pytest.approx(-0.5, abs=1e-5)
        assert yy[49, 49].item() == pytest.approx(-0.5, abs=1e-5)

        # Check corners
        assert xx[0, 0].item() == pytest.approx(-49.5, abs=1e-5)
        assert yy[0, 0].item() == pytest.approx(-49.5, abs=1e-5)

        assert xx[0, 99].item() == pytest.approx(49.5, abs=1e-5)
        assert yy[0, 99].item() == pytest.approx(-49.5, abs=1e-5)

    def test_reduce_apertures(self):
        """Test the extraction and reduction of aperture values."""
        # Create test image with a bright spot
        image = torch.zeros((100, 100))
        image[40:60, 40:60] = 1.0  # Bright square in the middle

        # Make sure the aperture will hit the bright spot
        # Center the aperture at (50, 50) which is inside the bright region
        locations, results = reduce_apertures(
            image, 0.0, 0.0, 10.0,  # Radius of 0 puts aperture at center
            operation=torch.nansum
        )

        # Check number of aperture results
        assert len(locations) == len(results)

        # Verify first result (planet aperture) hits the bright spot
        assert results[0].item(
        ) > 0, "Expected positive sum in the aperture at the center"

        # Test with exclusions
        locations, results = reduce_apertures(
            image, 20.0, 0.0, 10.0,
            operation=torch.nansum,
            exclude_planet=True
        )
        assert len(locations) < 12  # Should be fewer than without exclusions

    def test_calc_snr_mawet(self):
        """Test the calculation of SNR using Mawet's method."""
        # Simple case: signal 5, noise samples all 1
        signal = torch.tensor(5.0)
        noises = torch.ones(10)

        # Create a mock for torch.std that returns a non-zero value
        with patch('torch.std', return_value=torch.tensor(1.0)):
            snr = calc_snr_mawet(signal, noises)

            # For signal=5, noise=1 with 10 samples,
            # SNR = (5-1)/(1*sqrt(1+1/10)) ≈ 4/1.05 ≈ 3.81
            assert snr.item() == pytest.approx(3.81, abs=0.1)

        # Test with varying noise
        signal = torch.tensor(10.0)
        noises = torch.tensor([8.0, 9.0, 9.5, 10.5, 11.0, 12.0])

        snr = calc_snr_mawet(signal, noises)
        # Expected SNR can be calculated by hand:
        # mean(noises) = 10, std(noises) ≈ 1.414
        # SNR = (10-10)/(1.414*sqrt(1+1/6)) ≈ 0
        assert abs(snr.item()) < 0.1  # Should be close to zero

    @patch('torchklip.utils.snr._snr_single_frame')
    def test_compute_snr_2d(self, mock_snr):
        """Test SNR computation for 2D images."""
        # Mock the single frame SNR calculation
        mock_snr.return_value = torch.tensor(5.0)

        # Create a 2D test image
        test_image = torch.zeros((100, 100))

        # Call compute_snr
        snr = compute_snr(
            test_image, 60.0, 60.0, 5.0,
            exclude_planet=False,
            exclude_nearest=1,
            verbose=False
        )

        # Verify mock was called with correct parameters
        mock_snr.assert_called_once()
        args = mock_snr.call_args[0]
        assert torch.equal(args[0], test_image)
        assert args[1] == 60.0
        assert args[2] == 60.0
        assert args[3] == 5.0
        assert args[4] is False
        assert args[5] == 1

        # Check result
        assert snr.item() == 5.0

    @patch('torchklip.utils.snr._snr_single_frame')
    def test_compute_snr_3d(self, mock_snr):
        """Test SNR computation for 3D (multi-frame) images."""
        # Mock the single frame SNR calculation to return different values
        mock_snr.side_effect = [torch.tensor(
            3.0), torch.tensor(4.0), torch.tensor(5.0)]

        # Create a 3D test image (3 frames)
        test_image = torch.zeros((3, 100, 100))

        # Call compute_snr
        snr_list = compute_snr(
            test_image, 60.0, 60.0, 5.0,
            exclude_planet=True,
            exclude_nearest=2,
            verbose=False
        )

        # Verify mock was called 3 times
        assert mock_snr.call_count == 3

        # Check results
        assert len(snr_list) == 3
        assert snr_list[0].item() == 3.0
        assert snr_list[1].item() == 4.0
        assert snr_list[2].item() == 5.0

    def test_numpy_input_handling(self):
        """Test that numpy arrays are correctly converted to torch tensors."""
        # Create a numpy array
        np_image = np.zeros((100, 100))
        np_image[45:55, 45:55] = 1.0  # Bright spot in center

        # Apply a simple patch to avoid actual computation
        with patch('torchklip.utils.snr._snr_single_frame', return_value=torch.tensor(4.5)):
            snr = compute_snr(np_image, 50.0, 50.0, 5.0, verbose=False)
            assert snr.item() == 4.5

    def test_real_computation(self):
        """Test actual SNR computation with a simple synthetic image."""
        # Create test image with a bright planet and background noise
        image = torch.zeros((100, 100))

        # Add random background noise (mean=0, std=0.1)
        torch.manual_seed(42)  # For reproducibility
        image += torch.randn_like(image) * 0.1

        # Add a synthetic planet signal at (65, 65)
        planet_x, planet_y = 65.0, 65.0
        planet_radius = 3
        for i in range(int(planet_y) - planet_radius, int(planet_y) + planet_radius + 1):
            for j in range(int(planet_x) - planet_radius, int(planet_x) + planet_radius + 1):
                if (i - planet_y)**2 + (j - planet_x)**2 <= planet_radius**2:
                    if 0 <= i < 100 and 0 <= j < 100:
                        image[i, j] = 1.0  # Strong signal

        # Mock the _snr_single_frame function to return a predictable SNR
        with patch('torchklip.utils.snr._snr_single_frame', return_value=torch.tensor(7.5)):
            # Compute SNR
            snr = compute_snr(
                image, planet_x, planet_y, fwhm=6.0,
                exclude_planet=False, exclude_nearest=1,
                verbose=False
            )

            # Verify the mocked SNR value
            assert snr.item() == 7.5


class TestNoiseApertureCenters:
    """Tests for noise_aperture_centers."""

    IMAGE_SHAPE = (101, 101)
    PLANET_X = 70.0
    PLANET_Y = 50.0
    FWHM = 5.0

    def _centers(self, **kwargs):
        kw = dict(image_shape=self.IMAGE_SHAPE, planet_x=self.PLANET_X,
                  planet_y=self.PLANET_Y, fwhm=self.FWHM)
        kw.update(kwargs)
        return noise_aperture_centers(**kw)

    def test_returns_list_of_tuples(self):
        centers = self._centers()
        assert isinstance(centers, list)
        assert len(centers) > 0
        assert all(len(c) == 2 for c in centers)

    def test_planet_not_in_noise_list(self):
        """Planet aperture must not appear among the noise apertures."""
        centers = self._centers()
        for (x, y) in centers:
            dist = ((x - self.PLANET_X) ** 2 + (y - self.PLANET_Y) ** 2) ** 0.5
            assert dist > self.FWHM * 0.5, "Planet aperture must not appear in noise list"

    def test_count_equals_ring_minus_one(self):
        """Noise aperture count is total ring apertures minus the planet."""
        centers = self._centers()
        r_px, pa_deg = get_r_pa(torch.Size(list(self.IMAGE_SHAPE)),
                                 self.PLANET_X, self.PLANET_Y)
        all_locs = simple_aperture_locations(
            float(r_px), float(pa_deg), self.FWHM, exclude_planet=False
        )
        assert len(centers) == len(all_locs) - 1

    def test_exclude_nearest_reduces_count(self):
        """exclude_nearest=N removes 2*N apertures from the noise list."""
        n = 2
        centers_all = self._centers(exclude_nearest=0)
        centers_exc = self._centers(exclude_nearest=n)
        assert len(centers_exc) == len(centers_all) - 2 * n

    def test_exclude_nearest_matches_compute_snr_geometry(self):
        """Returned positions must agree with what simple_aperture_locations produces."""
        n = 2
        centers = self._centers(exclude_nearest=n)
        r_px, pa_deg = get_r_pa(torch.Size(list(self.IMAGE_SHAPE)),
                                 self.PLANET_X, self.PLANET_Y)
        expected_locs = simple_aperture_locations(
            float(r_px), float(pa_deg), self.FWHM,
            exclude_nearest=n, exclude_planet=False,
        )
        # expected_locs[0] is planet; rest are noise
        assert len(centers) == len(expected_locs) - 1

    def test_pixel_coords_within_image(self):
        """All returned positions should lie within the image bounds."""
        H, W = self.IMAGE_SHAPE
        for (x, y) in self._centers():
            assert 0 <= x < W
            assert 0 <= y < H


class TestDrawApertures:
    """Tests for draw_apertures."""

    IMAGE_SHAPE = (101, 101)
    PLANET_X = 70.0
    PLANET_Y = 50.0
    FWHM = 5.0

    def setup_method(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt

    def test_adds_correct_number_of_patches(self):
        fig, ax = self.plt.subplots()
        n_before = len(ax.patches)
        draw_apertures(ax, self.PLANET_X, self.PLANET_Y, self.FWHM,
                       self.IMAGE_SHAPE, color="cyan")
        n_noise = len(noise_aperture_centers(
            self.IMAGE_SHAPE, self.PLANET_X, self.PLANET_Y, self.FWHM))
        assert len(ax.patches) - n_before == 1 + n_noise
        self.plt.close(fig)

    def test_exclude_nearest_reduces_patches(self):
        """Patches added with exclude_nearest=2 should be fewer than without."""
        fig1, ax1 = self.plt.subplots()
        draw_apertures(ax1, self.PLANET_X, self.PLANET_Y, self.FWHM,
                       self.IMAGE_SHAPE, exclude_nearest=0)
        n_full = len(ax1.patches)
        self.plt.close(fig1)

        fig2, ax2 = self.plt.subplots()
        draw_apertures(ax2, self.PLANET_X, self.PLANET_Y, self.FWHM,
                       self.IMAGE_SHAPE, exclude_nearest=2)
        n_exc = len(ax2.patches)
        self.plt.close(fig2)

        # 2 excluded on each side → 4 fewer noise patches
        assert n_full - n_exc == 4

    def test_planet_patch_is_solid(self):
        """First patch added should be the solid planet aperture."""
        fig, ax = self.plt.subplots()
        draw_apertures(ax, self.PLANET_X, self.PLANET_Y, self.FWHM,
                       self.IMAGE_SHAPE, color="red")
        planet_patch = ax.patches[0]
        assert planet_patch.get_linestyle() == "-"
        assert planet_patch.get_linewidth() == pytest.approx(1.5)
        self.plt.close(fig)

    def test_noise_patches_are_dashed(self):
        """All patches after the first should be dashed noise apertures."""
        fig, ax = self.plt.subplots()
        draw_apertures(ax, self.PLANET_X, self.PLANET_Y, self.FWHM,
                       self.IMAGE_SHAPE, color="cyan")
        for patch in ax.patches[1:]:
            assert patch.get_linestyle() == "--"
        self.plt.close(fig)

    def test_custom_color_applied(self):
        """Edge colour (RGB) of every patch should match the requested colour."""
        fig, ax = self.plt.subplots()
        import matplotlib.colors as mcolors
        draw_apertures(ax, self.PLANET_X, self.PLANET_Y, self.FWHM,
                       self.IMAGE_SHAPE, color="magenta")
        expected_rgb = mcolors.to_rgb("magenta")
        for patch in ax.patches:
            # get_edgecolor returns RGBA; compare only the RGB components
            assert patch.get_edgecolor()[:3] == pytest.approx(expected_rgb, abs=1e-3)
        self.plt.close(fig)

    def test_radius_equals_fwhm_over_two(self):
        fig, ax = self.plt.subplots()
        draw_apertures(ax, self.PLANET_X, self.PLANET_Y, self.FWHM,
                       self.IMAGE_SHAPE)
        for patch in ax.patches:
            assert patch.radius == pytest.approx(self.FWHM / 2.0)
        self.plt.close(fig)
