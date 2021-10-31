import numpy as np
from typing import List, Tuple, Union, Callable

import torch
import torch.nn.functional as F

import fastmri
from fastmri import complex_abs, complex_mul, complex_conj, fft2c, ifft2c
from fastmri.data import transforms as T

from kernprior import utils


def chan_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Move the real and imaginary part of a complex-valued tensor from the channel dimension to a separate dimension.

    Args:
        x: Tensor of shape (B, C*2, H, W).

    Returns:
        A tensor of shape (B, C, H, W, 2), where the real and imaginary part are stacked in the last dimension.

    Notes:
        Based on NormUnet from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/models/varnet.py
    """
    B, C2, H, W = x.shape
    assert C2 % 2 == 0
    C = C2 // 2
    x = x.reshape(B, 2, C, H, W)
    return x.permute(0, 2, 3, 4, 1)


def complex_to_chan(x: torch.Tensor) -> torch.Tensor:
    """
    Move the real and imaginary part of a complex-valued tensor from a separate dimension to the channel dimension.

    Args:
        x: Tensor of shape (B, C, H, W, 2), where the last dimension is the complex dimension.

    Returns:
        A tensor of shape (B, C*2, H, W), with the complex dimension as part of the channel dimension.

    Notes:
        Based on NormUnet from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/models/varnet.py
    """
    B, C, H, W, two = x.shape
    assert two == 2
    x = x.permute(0, 4, 1, 2, 3)
    return x.reshape(B, 2 * C, H, W)


def to_polar(x: torch.Tensor) -> torch.Tensor:
    """
    Transform a complex-valued tensor from cartesian to polar coordinates.

    Args:
        x: Tensor whose last dimension contains the real and imaginary part.

    Returns:
        x in polar coordinates. x[...,0] contains the magnitude and x[...,1] contains the phase.
    """
    r = complex_abs(x)
    phi = torch.atan2(x[...,1], x[...,0])
    return torch.stack((r, phi), dim=-1)


def to_cartesian(x: torch.Tensor) -> torch.Tensor:
    """
    Transform a complex-valued tensor from polar to cartesian coordinates.

    Args:
        x: Tensor whose last dimension contains the magnitude and phase.

    Returns:
        x in cartesian coordinates. x[...,0] contains the real part and x[...,1] contains the imaginary part.
    """
    r = x[...,0]
    phi = x[...,1]
    real = r * phi.cos()
    imag = r * phi.sin()
    return torch.stack((real, imag), dim=-1)


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def generate(
    zs: List[torch.Tensor],
    glow,
    height: int,
    width: int,
    overlap: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """
    Generate an image based on a batch of latent vectors using a trained generative flow.

    Args:
        zs: The latent vectors that get introduced into the generative model at different levels. The patches are
            stacked in the batch dimension and the remaining dimensions are determined by the architecture of the model.
        glow: The model used to generate image patches based on the latent vectors.
        height: The height of the output image.
        width: The width of the output image.
        overlap: The overlap between adjacent patches.

    Returns:
        A complex image of shape (height, width, 2).
    """
    x, _ = glow.inverse(zs)
    x = compose(x, height, width, overlap)
    x = x.unsqueeze(0)
    x = chan_to_complex(x)
    x = x.squeeze(0).squeeze(0)
    return x


def compose(
    x: torch.Tensor,
    height: int,
    width: int,
    overlap: Union[int, Tuple[int, int]]
) -> torch.Tensor:
    """
    Combine patches into an image. At locations where multiple patches overlap, the output is the mean of the pixel values.

    Args:
        x: Tensor of shape (n_patches, C, patch_H, patch_W).
        height: The height of the output image.
        width: The width of the output image.
        overlap: The overlap between adjacent patches.

    Returns:
        An image of shape (C, height, width).
    """
    unnormalized_res = compose_additive(x, height, width, overlap)
    normalization_map = compose_additive(torch.ones_like(x), height, width, overlap)
    return unnormalized_res / normalization_map


def compose_additive(
    x: torch.Tensor,
    height: int,
    width: int,
    overlap: Union[int, Tuple[int, int]]
) -> torch.Tensor:
    """
    Combine patches into an image. At locations where multiple patches overlap, the output is the sum of the pixel values.

    Args:
        x: Tensor of shape (n_patches, C, patch_H, patch_W).
        height: The height of the output image.
        width: The width of the output image.
        overlap: The overlap between adjacent patches.

    Returns:
        An image of shape (C, height, width).
    """
    patch_size = x.shape[-2:]
    grid_shape, output_size, patch_size, stride = get_compose_params(height, width, patch_size, overlap)
    assert x.shape[0] == np.prod(grid_shape), 'invalid number of input patches'

    x_foldable = x.reshape(x.shape[0], -1).t().unsqueeze(0)
    x_folded = F.fold(x_foldable,
                      output_size=output_size,
                      kernel_size=patch_size,
                      stride=stride)
    x_folded = x_folded.squeeze(0)
    return T.center_crop(x_folded, (height, width))


def get_compose_params(
    height: int,
    width: int,
    patch_size: Union[int, Tuple[int, int]],
    overlap: Union[int, Tuple[int, int]]
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Calculate the number of patches needed to generate an image with the given constraints.

    Args:
        height: The desired height.
        width: The desired width.
        patch_size: The desired patch size.
        overlap: The desired overlap between adjacent patches.

    Returns:
        tuple containing:
            grid_shape: The number of patches needed in each spatial dimension.
            output_size: The smallest height and width that fulfill the constraints given as inputs,
                i.e. output_size[0] >= height and output_size[1] >= width.
            patch_size: If the input patch_size is already a tuple, it is returned unchanged.
                If patch_size is an int, it is expanded to apply to both spatial dimensions.
            stride: The stride of the sliding blocks, i.e. patch_size - overlap.
    """
    height_width = torch.tensor((height, width)).expand(2)
    patch_size = torch.tensor(patch_size).expand(2)
    overlap = torch.tensor(overlap).expand(2)

    stride = patch_size - overlap
    assert (stride > 0).all()
    # this is easiest to understand by first assuming overlap == 0, i.e. stride == patch_size and then generalizing
    grid_shape = ((height_width - patch_size) / stride + 1).ceil().to(int)
    assert (grid_shape > 0).all()
    output_size = patch_size + (grid_shape - 1) * stride
    assert (output_size >= height_width).all()

    return (utils.to_tuple_of_int(grid_shape),  # type: ignore
            utils.to_tuple_of_int(output_size),
            utils.to_tuple_of_int(patch_size),
            utils.to_tuple_of_int(stride))


def crop_spatially(kspace: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
    """
    Center crop the kspace spatially, i.e. take the IFFT of the input kspace, center crop to the specified crop size
        and apply the FFT to go back to the frequency domain.

    Args:
        kspace: Tensor of shape (..., H, W, 2), where ... mean an arbitrary number of leading dimensions and the last
            dimension is the complex dimension.
        crop_size: The spatial dimensions to crop to.

    Returns:
        The cropped kspace.
    """
    image = ifft2c(kspace)
    image = T.complex_center_crop(image, crop_size)
    return fft2c(image)


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    """Normalize the image to [0, 1] by subtracting the min and dividing by the max."""
    x = x - x.min()
    return x / x.max()


def normalize_kspace(kspace: torch.Tensor, challenge: str) -> torch.Tensor:
    """
    Normalize the kspace by dividing each pixel by the maximum of the magnitude of the corresponding image.
        In case of multi-coil input, this corresponds to the maximum pixel value of the RSS image.

    Args:
        kspace: kspace of shape (num_coils, H, W, 2) for multi-coil data or (H, W, 2) for single-coil data.
        challenge: One of ('singlecoil', 'multicoil').

    Returns:
        The normalized kspace.
    """
    return kspace / get_image(kspace, challenge).max()


def get_image(kspace: torch.Tensor, challenge: str) -> torch.Tensor:
    """
    Get the magnitude of the image corresponding to the input kspace. In case of multi-coil input,
        this corresponds to the RSS estimate.

    Args:
        kspace: kspace of shape (num_coils, H, W, 2) for multi-coil data or (H, W, 2) for single-coil data.
        challenge: One of ('singlecoil', 'multicoil').

    Returns:
        The corresponding image of shape (H, W).
    """
    image = ifft2c(kspace)
    image = complex_abs(image)
    if challenge == 'multicoil':
        image = fastmri.rss(image)
    return image


def get_forward_model(
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    challenge: str,
    sens_params: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return a function that applies the forward model of accelerated MRI to an input image. In case challenge == 'multicoil',
        sensitivity maps are estimated using ESPIRiT.

    Args:
        masked_kspace: Undersampled kspace of shape (num_coils, H, W, 2) for multi-coil data or (H, W, 2) for single-coil data.
        mask: mask of shape (1, 1, W, 1) for multi-coil data or (1, W, 1) for single-coil data.
        challenge: One of ('singlecoil', 'multicoil').
        sens_params: Parameters that get directly passed to ecalib from the Bart toolbox.
            Only relevant if challenge == 'multicoil'. See utils.get_sens_maps for details.

    Returns:
        A function that takes an image of shape (H, W, 2) as input and applies the MRI forward model to it.
            The output of the forward model is of shape (num_coils, H, W, 2) if challenge == 'multicoil' or
            (H, W, 2) if challenge == 'singlecoil'.
    """
    if challenge == 'multicoil':
        assert masked_kspace.ndim == mask.ndim == 4
        sens_maps = utils.get_sens_maps(masked_kspace, sens_params).to(masked_kspace.device)

        def forward_model(x):
            return mask * fft2c(complex_mul(x, sens_maps))
    else:
        assert mask.ndim == 3

        def forward_model(x):
            return mask * fft2c(x)

    return forward_model


def data_consistency(
    x: torch.Tensor,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    sens_params: str,
    challenge: str,
) -> torch.Tensor:
    """
    Enforce data consistency by replacing frequencies of x with those of masked_kspace where the mask is nonzero.

    Args:
        x: The complex-valued image of shape (H, W, 2)
        masked_kspace: Undersampled kspace of shape (num_coils, H, W, 2) for multi-coil data or (H, W, 2) for single-coil data.
        mask: mask of shape (1, 1, W, 1) for multi-coil data or (1, W, 1) for single-coil data.
        sens_params: Parameters that get directly passed to ecalib from the Bart toolbox.
            Only relevant if challenge == 'multicoil'. See utils.get_sens_maps for details.
        challenge: One of ('singlecoil', 'multicoil').

    Returns:
        The magnitude of the resulting image of shape (H, W).
    """
    if challenge == 'multicoil':
        sens_maps = utils.get_sens_maps(masked_kspace, sens_params).to(x.device)
        x = sens_expand(x, sens_maps)

    y = fft2c(x)
    mask = mask.expand(masked_kspace.shape).bool()
    y[mask] = masked_kspace[mask]
    x = ifft2c(y)

    if challenge == 'multicoil':
        x = fastmri.rss_complex(x)
    else:
        x = complex_abs(x)

    return x


def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Combine num_coils individual coil images into a single image using estimates of the sensitivity maps.
        This is the inverse of sens_expand.

    Args:
        x: Tensor of shape (num_coils, H, W, 2) that contains the individual coil images.
        sens_maps: Sensitivity maps of shape (num_coils, H, W, 2).

    Returns:
        The combined image of shape (H, W, 2).

    Notes:
        Based on VarNetBlock from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/models/varnet.py
    """
    return complex_mul(x, complex_conj(sens_maps)).sum(dim=0)


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Expand a single image into num_coils individual coil images using estimates of the sensitivity maps.
        This is the inverse of sens_reduce.

    Args:
        x: An image of shape (H, W, 2).
        sens_maps: Sensitivity maps of shape (num_coils, H, W, 2).

    Returns:
        A tensor of shape (num_coils, H, W, 2) that contains the individual coil images.

    Notes:
        Based on VarNetBlock from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/models/varnet.py
    """
    return complex_mul(x, sens_maps)


def sens_reduce_precise(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """
    Combine num_coils individual coil images into a single image using estimates of the sensitivity maps.
        Then replace the magnitude with the RSS estimate to reduce inaccuracies caused by the sensitivity map estimates.

    Args:
        x: Tensor of shape (num_coils, H, W, 2) that contains the individual coil images.
        sens_maps: Sensitivity maps of shape (num_coils, H, W, 2).

    Returns:
        The combined image of shape (H, W, 2).
    """
    x_reduced = sens_reduce(x, sens_maps)
    x_phi = to_polar(x_reduced)[...,1]
    x_rss = fastmri.rss_complex(x)
    x_polar = torch.stack((x_rss, x_phi), dim=-1)
    return to_cartesian(x_polar)
