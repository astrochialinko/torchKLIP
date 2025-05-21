# torchklip/utils/snr.py
from typing import Tuple, List, Callable, Union

import torch
import numpy as np

# local modules
from .logging_utils import get_logger

# Get a logger specific to this module
logger = get_logger(__name__.split('.')[-1])


def get_r_pa(image_shape: torch.Size, planet_x: float, planet_y: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates of a planet to polar coordinates (radius and position angle).

    Args:
        image_shape (torch.Size): Shape of the image as (H, W).
        planet_x (float): X-coordinate of the planet in pixels.
        planet_y (float): Y-coordinate of the planet in pixels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            r_px: Radial distance from the image center in pixels.
            pa_deg: Position angle in degrees east of north.
    """
    # Compute image center coordinates
    height, width = image_shape
    ctr_x = (width - 1.) / 2.
    ctr_y = (height - 1.) / 2.

    # Offsets from center
    dx = torch.tensor(planet_x - ctr_x, dtype=torch.float32)
    dy = torch.tensor(planet_y - ctr_y, dtype=torch.float32)

    # Radial distance
    r_px = torch.hypot(dx, dy)

    # Angle in radians, then convert to degrees and shift to east-of-north convention
    # atan2 returns angle in radians
    pa_rad = torch.atan2(dy, dx)
    # shift so that 0 deg is north
    pa_deg = pa_rad.mul(180.0 / np.pi).sub(90.0)

    return r_px, pa_deg


def simple_aperture_locations(r_px: float, pa_deg: float, resolution_element_px: float,
                              exclude_nearest: int = 0, exclude_planet: bool = False) -> List[Tuple[float, float]]:
    """
    Compute aperture center offsets arranged in a ring at a given radius.

    Args:
        r_px (float): Radial distance from the center in pixels.
        pa_deg (float): Starting position angle in degrees east of north.
        resolution_element_px (float): Aperture diameter in pixels.
        exclude_nearest (int, optional): Number of apertures to skip on either side of the planet. Defaults to 0.
        exclude_planet (bool, optional): If True, exclude the planet aperture. Defaults to False.

    Returns:
        List[Tuple[float, float]]: List of (offset_x, offset_y) pixel positions for each aperture.
    """
    # Total ring length
    circumference = 2 * np.pi * r_px
    # Number of apertures based on resolution element
    n_apertures = max(1, int(circumference / resolution_element_px))

    # Starting angle (convert degrees to radians, add 90 deg for frame rotation)
    start_theta = np.deg2rad(pa_deg + 90)
    delta_theta = np.deg2rad(360 / n_apertures)
    locations = []

    # Include planet aperture if exclude_planet is False
    if not exclude_planet:
        locations.append((r_px * np.cos(start_theta),
                         r_px * np.sin(start_theta)))

    # Append remaining apertures around the ring, skipping nearest if requested
    for i in range(1 + exclude_nearest, n_apertures - exclude_nearest):
        offset_x = r_px * np.cos(start_theta + i * delta_theta)
        offset_y = r_px * np.sin(start_theta + i * delta_theta)
        locations.append((offset_x, offset_y))

    return locations


def cartesian_coords(center: Tuple[float, float], data_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Cartesian coordinate grids for an image.

    Args:
        center (Tuple[float, float]): (x, y) coordinates of the image center.
        data_shape (torch.Size): Shape of the image as (H, W).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            xx: X-coordinate grid offset by center_x.
            yy: Y-coordinate grid offset by center_y.
    """
    # Create integer grids then shift by center to get x,y offsets
    height, width = data_shape
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(
        width, dtype=torch.float32), indexing='ij'
    )
    center_x, center_y = center
    xx = xx - center_x  # shift x-grid
    yy = yy - center_y  # shift y-grid
    return xx, yy


def reduce_apertures(image: torch.Tensor,
                     r_px: float,
                     starting_pa_deg: float,
                     resolution_element_px: float,
                     operation: Callable[[torch.Tensor], torch.Tensor],
                     exclude_nearest: int = 0,
                     exclude_planet: bool = False) -> Tuple[List[Tuple[float, float]], List[torch.Tensor]]:
    """
    Extract values inside circular apertures arranged in a ring around the image center.

    Args:
        image (torch.Tensor): 2D input image tensor.
        r_px (float): Radius for the ring of apertures in pixels.
        starting_pa_deg (float): Starting position angle in degrees east of north.
        resolution_element_px (float): Aperture diameter in pixels.
        operation (Callable): Function applied to aperture pixels (e.g., torch.nanmedian).
        exclude_nearest (int, optional): Number of adjacent apertures to skip. Defaults to 0.
        exclude_planet (bool, optional): If True, exclude the planet aperture. Defaults to False.

    Returns:
        Tuple[List[Tuple[float, float]], List[torch.Tensor]]:
            locations: List of (offset_x, offset_y) positions for each aperture.
            results: List of computed values for each aperture.
    """
    # If image has extra dims (e.g., channel), flatten to first frame
    if image.ndim > 2:
        image = image.reshape(-1, image.shape[-2], image.shape[-1])[0]
    height, width = image.shape

    # Compute center and coordinate grids
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    xx, yy = cartesian_coords(center, image.shape)

    # Determine aperture centers
    locations = simple_aperture_locations(r_px, starting_pa_deg, resolution_element_px,
                                          exclude_nearest=exclude_nearest, exclude_planet=exclude_planet)
    aperture_radius = resolution_element_px / 2.0
    results = []

    # Loop through each aperture location
    for (offset_x, offset_y) in locations:

        # Create a mask for the circular aperture
        dist = torch.sqrt((xx - offset_x) ** 2 + (yy - offset_y) ** 2)
        mask = dist <= aperture_radius

        # Apply reduction operation to masked pixels
        results.append(operation(image[mask]))

    return locations, results


def calc_snr_mawet(signal: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
    """
    Compute the signal-to-noise ratio using the two-sample t-test formulation (Mawet et al. 2014).

    Args:
        signal (torch.Tensor): Measured signal from the planet aperture.
        noises (torch.Tensor): Noise measurements from surrounding apertures.

    Returns:
        torch.Tensor: Computed SNR.
    """
    # Flatten noise apertures into 1D
    noises = noises.flatten()

    # (signal - mean_noise) / (std_noise * sqrt(1 + 1/N))
    return (signal - torch.mean(noises)) / (
        torch.std(noises) *
        torch.sqrt(1 + 1 / torch.tensor(len(noises), dtype=torch.float32))
    )


def _snr_single_frame(
    frame: torch.Tensor,
    planet_x: float,
    planet_y: float,
    fwhm: float,
    exclude_planet: bool,
    exclude_nearest: int,
    operation: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
    Compute SNR for a single image frame.

    Args:
        frame (torch.Tensor): 2D image tensor for one frame.
        planet_x (float): X-coordinate of the planet in pixels.
        planet_y (float): Y-coordinate of the planet in pixels.
        fwhm (float): Aperture diameter (pixels) for noise estimation.
        exclude_planet (bool): If True, exclude the planet aperture from noise.
        exclude_nearest (int): Number of nearest apertures to skip.
        operation (Callable): Reduction function for aperture values.

    Returns:
        torch.Tensor: SNR value for the frame.
    """

    # Get radius and angle
    image_shape = frame.shape[-2:]
    r_px, pa_deg = get_r_pa(image_shape, planet_x, planet_y)

    # Extract aperture values
    locations, results = reduce_apertures(
        frame, r_px, pa_deg, fwhm,
        operation=operation,
        exclude_nearest=exclude_nearest,
        exclude_planet=exclude_planet
    )

    # Planet signal is the first aperture
    planet_signal = results[0]

    # Remaining apertures are noise samples
    noise_vals = torch.tensor(
        [res for res in results[1:]], dtype=torch.float32)

    return calc_snr_mawet(planet_signal, noise_vals)


def compute_snr(klip_image: Union[torch.Tensor, np.ndarray],
                planet_x: float,
                planet_y: float,
                fwhm: float,
                exclude_planet: bool = False,
                exclude_nearest: int = 0,
                operation: Callable[[torch.Tensor],
                                    torch.Tensor] = torch.nanmedian,
                verbose: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Compute the SNR of a planet in a KLIP-processed image using PyTorch operations.
    Works with both 2D and 3D images. For 3D images, returns a list of SNR values for each frame.

    Args:
        klip_image (torch.Tensor or np.ndarray): The KLIP-processed image (2D or 3D).
        planet_x (float): x-coordinate (pixels) of the planet.
        planet_y (float): y-coordinate (pixels) of the planet.
        fwhm (float): Aperture diameter in pixels (e.g., full width at half maximum).
        exclude_planet (bool, optional): If True, do not include the planet aperture in noise estimation.
        exclude_nearest (int, optional): Number of adjacent apertures to exclude.
        operation (Callable, optional): Operation to apply within each aperture (default: torch.nanmedian).
        verbose(bool, optional): If True, print the SNR result.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]:
            - For 2D images: A scalar tensor representing the SNR.
            - For 3D images: A list of SNR values for each frame.
    """
    # Convert input image to a torch.Tensor if it is a numpy array
    if isinstance(klip_image, np.ndarray):
        klip_image = torch.from_numpy(klip_image)

    # Single-frame case
    if klip_image.ndim == 2:
        snr = _snr_single_frame(
            klip_image, planet_x, planet_y, fwhm,
            exclude_planet, exclude_nearest, operation
        )
        if verbose:
            logger.info(f"SNR: {snr.item():.2f}")
        return snr

    # Multi-frame (3D) case: loop over frames
    snr_list = []
    for idx in range(klip_image.shape[0]):
        snr = _snr_single_frame(
            klip_image[idx], planet_x, planet_y, fwhm,
            exclude_planet, exclude_nearest, operation
        )
        if verbose:
            logger.info(f"Frame {idx}: SNR = {snr.item():.2f}")
        snr_list.append(snr)

    # Log summary statistics if requested
    if verbose:
        tensor = torch.tensor(snr_list)
        logger.info("====================================")
        logger.info(" Signal-to-Noise Ratio (SNR) Summary")
        logger.info("====================================")
        logger.info(
            f"Min SNR: {tensor.min().item():.2f} at Frame {tensor.argmin().item()}")
        logger.info(
            f"Max SNR: {tensor.max().item():.2f} at Frame {tensor.argmax().item()}")
        logger.info("------------------------------------\n")

    return snr_list


__all__ = ["compute_snr"]
