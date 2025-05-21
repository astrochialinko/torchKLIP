# torchklip/utils/image_plot.py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# local modules
from ..utils.logging_utils import get_logger

# Get a logger specific to this module
logger = get_logger(__name__.split('.')[-1])


def plot_image(image_data, center_coords=None, vmin=None, vmax=None,
               title=None, iwa_radius=None, owa_radius=None,
               scalebar_length=None, scalebar_width=0.3, scalebar_color='white', scalebar_label=None,
               xlim_half_range=None, ylim_half_range=None, figsize=(6, 6),
               colorbar_label='Counts', log_scale=False, show_center=False,
               xlabel="$x$ pixel", ylabel="$y$ pixel", output_filename=None,
               show_legend=False, **kwargs):
    """
    Plot image data with various annotations such as Inner Working Angle (IWA),
    center marker, scalebar, and colorbar.

    Parameters:
    -----------
    image_data : 2D numpy array
        The image data to be plotted
    center_coords : tuple or None, optional
        (x, y) coordinates of the center of the image, if None, uses center of image
    vmin, vmax : float, optional
        Minimum and maximum values for color scaling
    title : str or None, optional
        Title of the plot, if None, no title is shown
    iwa_radius : float or None, optional
        Radius of the Inner Working Angle circle in pixels, if None, no circle is shown
    owa_radius : float or None, optional
        Radius of the Outer Working Angle circle in pixels, if None, no circle is shown
    scalebar_length : float or None, optional
        Length of the scalebar in pixels, if None, no scalebar is shown
    scalebar_width : float or None, optional
        Width of the scalebar in pixels, if None, uses default width
    scalebar_color : str, optional
        Color of the scalebar, default 'white'
    scalebar_label : str or None, optional
        Label for the scalebar, if None, no scalebar is shown
    xlim_half_range : int or None, optional
        Half-width of the plot window in pixels, if None, shows full image width
    ylim_half_range : int or None, optional
        Half-height of the plot window in pixels, if None, shows full image height
    figsize : tuple, optional
        Figure size in inches, default (6, 6)
    colorbar_label : str, optional
        Label for the colorbar
    log_scale : bool, optional
        Whether to use logarithmic color scaling, default False
    show_center : bool, optional
        Whether to mark the center with a star
    xlabel : str, optional
        Label for the x-axis, default "$x$ pixel"
    ylabel : str, optional
        Label for the y-axis, default "$y$ pixel"
    output_filename : str or None, optional
        Filename to save the figure, if None, figure is not saved. 
        Default is None. If provided without extension, saves as PNG.
    show_legend : bool, optional
        Whether to show a legend for IWA and OWA circles, default False
    kwargs : 
        Additional keyword arguments passed to matplotlib.pyplot.imshow()

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(1, figsize=figsize)

    # If center_coords is None, use the center of the image
    center_coords = center_coords or (
        (image_data.shape[1] - 1) / 2, (image_data.shape[0] - 1) / 2)

    # Display the image data with appropriate normalization
    if log_scale:
        im = ax.imshow(image_data, origin='lower', norm=LogNorm(
            vmin=vmin, vmax=vmax), **kwargs)
    else:
        im = ax.imshow(image_data, origin='lower',
                       vmin=vmin, vmax=vmax, **kwargs)

    # Create a list to store legend handles and labels
    legend_elements = []

    # Add Inner Working Angle circle if iwa_radius is provided
    if iwa_radius is not None:
        iwa_circle = Circle(
            center_coords,
            iwa_radius,  # Already in pixels
            facecolor='none',
            edgecolor='w',
            lw=2,
            linestyle="--",
            alpha=0.5
        )
        ax.add_patch(iwa_circle)
        # Add to legend elements
        legend_elements.append(plt.Line2D([0], [0], color='w', lw=2, linestyle='--',
                                          label=f'IWA: {iwa_radius} px'))

    # Add Outer Working Angle circle if owa_radius is provided
    if owa_radius is not None:
        owa_circle = Circle(
            center_coords,
            owa_radius,  # Already in pixels
            facecolor='none',
            edgecolor='w',
            lw=2,
            linestyle="-.",
            alpha=0.5
        )
        ax.add_patch(owa_circle)
        # Add to legend elements
        legend_elements.append(plt.Line2D([0], [0], color='w', lw=2, linestyle='-.',
                                          label=f'OWA: {owa_radius} px'))

    # Add legend if requested and if there are elements to show
    if show_legend and legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)

    # Mark the center if show_center is True
    if show_center:
        ax.plot(center_coords[0], center_coords[1],
                '*', markersize=12, color='black')

    # Create properly sized colorbar with divider
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, fontsize=11)

    # Add scalebar only if both scalebar_length and scalebar_label are provided
    if scalebar_length is not None and scalebar_label is not None:
        fontprops = fm.FontProperties(size=10)
        scalebar = AnchoredSizeBar(
            ax.transData,
            scalebar_length,
            scalebar_label,
            'lower right',
            pad=0.5,
            sep=5,
            color=scalebar_color,
            frameon=False,
            size_vertical=scalebar_width,
            fontproperties=fontprops
        )
        ax.add_artist(scalebar)
    elif scalebar_length is not None or scalebar_label is not None:
        logger.warning(
            "Warning: Both scalebar_length and scalebar_label must be provided to plot the scale bar.")

    # Set title if provided
    if title is not None:
        ax.set_title(title, fontsize=16)

    # Set axis limits if ranges are provided
    if xlim_half_range is not None:
        ax.set_xlim([center_coords[0] - xlim_half_range,
                    center_coords[0] + xlim_half_range])
    if ylim_half_range is not None:
        ax.set_ylim([center_coords[1] - ylim_half_range,
                    center_coords[1] + ylim_half_range])

    # Set axis labels
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    # Ensure tight layout
    plt.tight_layout()

    # Save figure if filename is provided
    if output_filename is not None:
        # Add .png extension if no extension is provided
        if '.' not in output_filename.split('/')[-1]:
            output_filename = output_filename + '.png'
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        logger.info(f"Figure saved to {output_filename}")

    return fig, ax


def plot_image_sequence(cube, frame_indices=None, **kwargs):
    """
    Plot multiple frames from a data cube, with the same formatting.

    Parameters:
    -----------
    cube : 3D numpy array
        Data cube containing multiple image frames
    frame_indices : list or None, optional
        List of frame indices to plot. If None, plots all frames.
    kwargs :
        Additional keyword arguments passed to plot_image

    Returns:
    --------
    List of (fig, ax) tuples for each frame
    """
    if frame_indices is None:
        frame_indices = range(cube.shape[0])

    results = []
    for idx in frame_indices:
        fig, ax = plot_image(cube[idx], **kwargs)
        results.append((fig, ax))

    return results


class ImagePlotter:
    """
    Class-based implementation for plotting images.
    Useful for more complex plotting scenarios where state needs to be maintained.
    """

    def __init__(self, center_coords=None, figsize=(6, 6)):
        """
        Initialize the plotter with default parameters.

        Parameters:
        -----------
        center_coords : tuple or None
            (x, y) coordinates of the center of the image, if None, uses center of image
        figsize : tuple
            Figure size in inches, default (6, 6)
        """
        self.center_coords = center_coords
        self.figsize = figsize

    def plot(self, image_data, vmin=None, vmax=None, title=None,
             iwa_radius=None, owa_radius=None, scalebar_length=None, scalebar_label=None,
             xlim_half_range=None, ylim_half_range=None, colorbar_label='Counts',
             log_scale=False, show_center=True,
             xlabel="$x$ pixel", ylabel="$y$ pixel", output_filename=None,
             show_legend=False, **kwargs):
        """
        Plot an image with annotations.

        Parameters are the same as plot_image function.

        Returns:
        --------
        fig, ax : matplotlib figure and axes objects
        """

        return plot_image(
            image_data,
            center_coords=self.center_coords,
            vmin=vmin,
            vmax=vmax,
            title=title,
            iwa_radius=iwa_radius,
            owa_radius=owa_radius,
            scalebar_length=scalebar_length,
            scalebar_label=scalebar_label,
            xlim_half_range=xlim_half_range,
            ylim_half_range=ylim_half_range,
            figsize=self.figsize,
            colorbar_label=colorbar_label,
            log_scale=log_scale,
            show_center=show_center,
            xlabel=xlabel,
            ylabel=ylabel,
            output_filename=output_filename,
            show_legend=show_legend,
            **kwargs
        )

    def plot_sequence(self, cube, frame_indices=None, **kwargs):
        """
        Plot multiple frames from a data cube.

        Parameters are the same as plot_image_sequence function.

        Returns:
        --------
        List of (fig, ax) tuples for each frame
        """
        return plot_image_sequence(cube, frame_indices,
                                   center_coords=self.center_coords,
                                   figsize=self.figsize,
                                   **kwargs)


__all__ = ["plot_image", "plot_image_sequence", "ImagePlotter"]
