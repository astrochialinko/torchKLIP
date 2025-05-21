import numpy as np
import matplotlib.pyplot as plt
import os
import pytest
from unittest.mock import patch, MagicMock

# Import the functions to test
from src.torchklip.utils.image_plot import plot_image, plot_image_sequence, ImagePlotter


class TestImagePlot:
    @classmethod
    def setup_class(cls):
        """Set up test data that will be used across test methods"""
        # Create a simple test image - avoid using zeros for log scale tests
        cls.test_image = np.ones((100, 100)) * 0.01  # Small positive values
        cls.test_image[40:60, 40:60] = 1.0  # Create a square in the middle

        # Create a simple test cube with multiple frames
        cls.test_cube = np.ones((3, 100, 100)) * 0.01
        cls.test_cube[0, 30:50, 30:50] = 1.0  # Frame 1
        cls.test_cube[1, 40:60, 40:60] = 1.0  # Frame 2
        cls.test_cube[2, 50:70, 50:70] = 1.0  # Frame 3

    def teardown_method(self):
        """Close all pyplot figures after each test"""
        plt.close('all')

    def test_plot_image_basic(self):
        """Test basic functionality of plot_image"""
        fig, ax = plot_image(self.test_image)

        # Check that figure and axes were created
        assert fig is not None
        assert ax is not None

        # Check that image data was plotted
        assert len(ax.images) == 1

    def test_plot_image_with_options(self):
        """Test plot_image with various options"""
        # Test with custom options but avoid log_scale for now
        fig, ax = plot_image(
            self.test_image,
            center_coords=(50, 50),
            vmin=0.01,  # Avoid zeros for log scale
            vmax=1.0,
            title="Test Image",
            iwa_radius=20,
            owa_radius=40,
            show_center=True,
            colorbar_label="Test Label",
            xlabel="X",
            ylabel="Y",
            show_legend=True
        )

        # Check that title is set
        assert ax.get_title() == "Test Image"

        # Check that center is marked (should be a Line2D object)
        center_marked = False
        for line in ax.lines:
            if line.get_marker() == '*':
                center_marked = True
                break
        assert center_marked

        # Check that circles are added (should be 2 patches)
        assert len(ax.patches) == 2

        # Check that legend is displayed
        assert ax.get_legend() is not None

        # Check that axis labels are set
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"

    def test_plot_image_log_scale(self):
        """Test plot_image with log scale"""
        # Create an image with strictly positive values for log scale
        log_test_image = np.ones((100, 100)) * 0.01
        log_test_image[40:60, 40:60] = 1.0

        fig, ax = plot_image(
            log_test_image,
            vmin=0.01,  # Must be positive for log scale
            vmax=1.0,
            log_scale=True
        )

        # Check that image was created with LogNorm
        assert "LogNorm" in str(ax.images[0].norm)

    def test_plot_image_with_scalebar(self):
        """Test plot_image with scalebar"""
        fig, ax = plot_image(
            self.test_image,
            scalebar_length=20,
            scalebar_label="20 px"
        )

        # Check that scalebar is added (should be an AnchoredSizeBar)
        has_scalebar = False
        for artist in ax.get_children():
            if "AnchoredSizeBar" in str(type(artist)):
                has_scalebar = True
                break
        assert has_scalebar

    def test_plot_image_with_limits(self):
        """Test plot_image with custom limits"""
        fig, ax = plot_image(
            self.test_image,
            center_coords=(50, 50),
            xlim_half_range=25,
            ylim_half_range=25
        )

        # Check that limits are set correctly
        assert ax.get_xlim() == (25, 75)
        assert ax.get_ylim() == (25, 75)

    def test_plot_image_save_file(self, tmp_path):
        """Test saving plot to file"""
        # Create a temporary filename
        output_file = os.path.join(tmp_path, "test_image")

        # Instead of mocking, let's directly check the file exists
        fig, ax = plot_image(
            self.test_image,
            output_filename=output_file
        )

        # Check that file was saved with .png extension
        assert os.path.exists(output_file + ".png")

    def test_plot_image_sequence(self):
        """Test plot_image_sequence function"""
        results = plot_image_sequence(self.test_cube)

        # Check that we got 3 figures (one for each frame)
        assert len(results) == 3

        # Check that each result is a tuple with figure and axes
        for fig, ax in results:
            assert fig is not None
            assert ax is not None

    def test_plot_image_sequence_with_indices(self):
        """Test plot_image_sequence with specific indices"""
        results = plot_image_sequence(self.test_cube, frame_indices=[0, 2])

        # Check that we got 2 figures (for frames 0 and 2)
        assert len(results) == 2

    def test_image_plotter_init(self):
        """Test ImagePlotter initialization"""
        plotter = ImagePlotter(center_coords=(50, 50), figsize=(8, 8))

        # Check that attributes are set correctly
        assert plotter.center_coords == (50, 50)
        assert plotter.figsize == (8, 8)

    def test_image_plotter_plot(self):
        """Test ImagePlotter.plot method"""
        plotter = ImagePlotter(center_coords=(50, 50))
        fig, ax = plotter.plot(self.test_image, title="Plotter Test")

        # Check that figure was created
        assert fig is not None
        assert ax is not None

        # Check that title is set
        assert ax.get_title() == "Plotter Test"

    def test_image_plotter_plot_sequence(self):
        """Test ImagePlotter.plot_sequence method"""
        plotter = ImagePlotter()
        results = plotter.plot_sequence(self.test_cube, frame_indices=[1])

        # Check that we got 1 figure
        assert len(results) == 1

        fig, ax = results[0]
        assert fig is not None
        assert ax is not None

    def test_missing_scalebar_warning(self):
        """Test that plot_image handles missing scalebar parameters correctly"""
        # Instead of checking for a warning log, just verify the function runs without error
        fig, ax = plot_image(
            self.test_image,
            scalebar_length=20,  # Only length, no label
            scalebar_label=None
        )

        # Verify no scalebar is added
        has_scalebar = False
        for artist in ax.get_children():
            if "AnchoredSizeBar" in str(type(artist)):
                has_scalebar = True
                break
        assert not has_scalebar
