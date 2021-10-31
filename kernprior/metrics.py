import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Union, Tuple
import torch

from kernprior import transforms


def ssim(target: Union[np.ndarray, torch.Tensor], pred: Union[np.ndarray, torch.Tensor], normalize_to: str) -> float:
    """Compute the structural similarity index (SSIM)."""
    target, pred = metrics_preprocess(target, pred, normalize_to)
    return structural_similarity(target, pred, data_range=target.max())  # type: ignore


def vif(target: Union[np.ndarray, torch.Tensor], pred: Union[np.ndarray, torch.Tensor], normalize_to: str) -> float:
    """Compute the visual information fidelity (VIF)."""
    target, pred = metrics_preprocess(target, pred, normalize_to)
    return visual_information_fidelity(target, pred, sigma_nsq=target.mean())


def psnr(target: Union[np.ndarray, torch.Tensor], pred: Union[np.ndarray, torch.Tensor], normalize_to: str) -> float:
    """Compute the peak signal-to-noise ratio (PSNR)."""
    target, pred = metrics_preprocess(target, pred, normalize_to)
    return peak_signal_noise_ratio(target, pred, data_range=target.max())  # type: ignore


def metrics_preprocess(
    target: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    normalize_to: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize target and pred for metric calculation.

    Args:
        target: The target image.
        pred: The predicted image.
        normalize_to: One of ('target', 'pred'). If normalize_to == 'target', pred is scaled such that its mean and std
            is equal to the mean and std of target. Similarly, if normalize_to == 'pred', target is scaled such that
            its mean and std is equal to the mean and std of pred.

    Returns:
        tuple containing:
            target: The normalized target image.
            pred: The normalized predicted image.
    """
    assert normalize_to in ('target', 'pred')

    target = transforms.to_numpy(target)
    pred = transforms.to_numpy(pred)

    if normalize_to == 'target':
        pred = (pred - pred.mean()) / pred.std()
        pred = pred * target.std() + target.mean()
    else:
        target = (target - target.mean()) / target.std()
        target = target * pred.std() + pred.mean()

    return target, pred


def visual_information_fidelity(ref: np.ndarray, dist: np.ndarray, sigma_nsq: Union[int, float] = 1, eps: float = 1e-10) -> float:
    """Implementation from https://github.com/aizvorski/video-quality/blob/master/vifp.py"""
    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = gaussian_filter(ref, sd)
            dist = gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = gaussian_filter(ref, sd)
        mu2 = gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    return num / den
