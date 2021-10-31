import os
import sys
from kernprior import paths

bart_path = paths.get_bart_path()
os.environ['TOOLBOX_PATH'] = bart_path
sys.path.append(os.path.join(bart_path, 'python'))

import h5py
import numpy as np
import shutil
import time
import xml.etree.ElementTree as etree
import yaml
from bart import bart  # noqa
from copy import deepcopy
from traceback import TracebackException
from typing import Dict, List, Optional, Tuple, Union
from urllib.request import urlopen

import torch

import fastmri
from fastmri.data import transforms as T
from fastmri.data.mri_data import et_query

from kernprior.glow import Glow

CropSize = Union[Optional[int], Tuple[Optional[int], Optional[int]]]


class dotdict(dict):
    """
    A dict that provides access via dot notation.

    Notes:
        Taken from https://stackoverflow.com/a/23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        """Taken from https://stackoverflow.com/a/49902096"""
        return dotdict(deepcopy(dict(self), memo=memo))  # type: ignore


def load_kspace_slice(file_path: str, slice_index: Optional[int] = None, as_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """
    Load a kspace slice from a HDF5 file.

    Args:
        file_path: File path.
        slice_index: Optional; Index of the slice to load. By default the mid-slice of the volume is loaded.
        as_tensor: Optional; If True, convert to torch.Tensor with complex dim as last dim. Otherwise return a complex np.ndarray.

    Returns:
        The kspace slice.
    """
    if slice_index is None:
        slice_index = get_num_slices(file_path) // 2
    with h5py.File(file_path, 'r') as hf:
        slice_kspace = hf['kspace'][slice_index]
    if as_tensor:
        slice_kspace = T.to_tensor(slice_kspace)
    return slice_kspace


def load_target(file_path: str, slice_index: Optional[int] = None, challenge: str = 'multicoil') -> np.ndarray:
    """
    Load the target from a HDF5 file.

    Args:
        file_path: File path.
        slice_index: Optional; Index of the slice to load. By default, the target for the mid-slice of the volume is loaded.
        challenge: Optional; One of ('singlecoil', 'multicoil'). By default, the RSS target is loaded.
            If challenge == 'singlecoil', the ESC target is loaded.

    Returns:
        The target.
    """
    if slice_index is None:
        slice_index = get_num_slices(file_path) // 2
    with h5py.File(file_path, 'r') as hf:
        return hf[f'reconstruction_{"rss" if challenge == "multicoil" else "esc"}'][slice_index]


def get_num_slices(file_path: str) -> int:
    """Return the number of slices of the kspace volume stored as a HDF5 file at file_path."""
    with h5py.File(file_path, 'r') as hf:
        return hf['kspace'].shape[0]


def get_mid_slice_indices(num_slices: int, mid_slice_range: int) -> List[int]:
    """
    Get the indices of the slices around the mid-slice image.

    Args:
        num_slices: The number of slices of the volume.
        mid_slice_range: The number of slices around the mid-slice image on each side for which we want the indices.

    Returns:
        The indices of the mid-slice +/- mid_slice_range.
    """
    mid_slice_index = num_slices // 2
    mid_slice_indices = list(range(mid_slice_index - mid_slice_range, mid_slice_index + mid_slice_range + 1))
    assert mid_slice_indices[0] >= 0
    return mid_slice_indices


def get_acquisition(file_path: str) -> str:
    """Return the acquisition type of the measurement stored as a HDF5 file at file_path."""
    with h5py.File(file_path, 'r') as hf:
        return from_confusing_terminology(hf.attrs['acquisition'])


def from_confusing_terminology(acquisition: str) -> str:
    if acquisition == 'CORPDFS_FBK':
        return 'excluding_fat'
    elif acquisition == 'CORPD_FBK':
        return 'including_fat'
    else:
        return acquisition


def get_attrs(file_path: str) -> Dict:
    """Get the attributes of the measurement stored as a HDF5 file at file_path."""
    with h5py.File(file_path, 'r') as hf:
        attrs = dict(hf.attrs)
    attrs.update(retrieve_metadata(file_path))
    return attrs


def retrieve_metadata(file_path: str) -> Dict:
    """
    Get metadata of the measurement stored as a HDF5 file at file_path.

    Notes:
        Taken from fastmri.data.mri_data.SliceDataset._retrieve_metadata
    """
    with h5py.File(file_path, 'r') as hf:
        et_root = etree.fromstring(hf['ismrmrd_header'][()])

        enc = ['encoding', 'encodedSpace', 'matrixSize']
        enc_size = (
            int(et_query(et_root, enc + ['x'])),
            int(et_query(et_root, enc + ['y'])),
            int(et_query(et_root, enc + ['z'])),
        )
        rec = ['encoding', 'reconSpace', 'matrixSize']
        recon_size = (
            int(et_query(et_root, rec + ['x'])),
            int(et_query(et_root, rec + ['y'])),
            int(et_query(et_root, rec + ['z'])),
        )

        lims = ['encoding', 'encodingLimits', 'kspace_encoding_step_1']
        enc_limits_center = int(et_query(et_root, lims + ['center']))
        enc_limits_max = int(et_query(et_root, lims + ['maximum'])) + 1

        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max

        acquisition = hf.attrs['acquisition']

    return {
        'padding_left': padding_left,
        'padding_right': padding_right,
        'encoding_size': enc_size,
        'recon_size': recon_size,
        'acquisition': acquisition,
    }


def get_sens_maps(kspace: torch.Tensor, sens_params: str = '') -> torch.Tensor:
    """
    Compute the sensitivity maps for the input kspace.

    Args:
        kspace: kspace of shape (num_coils, H, W, 2).
        sens_params: Optional; Parameters that get directly passed to ecalib from the Bart toolbox.
            (for example '-W' to use soft-weighting of the singular vectors,
            '-c0' to crop the sensitivities if the eigenvalue is smaller than 0 or '-W -c0' to do both of these).
            See documentation of the Bart toolbox for details on the parameters that can be passed.

    Returns:
        The sensitivity maps of shape (num_coils, H, W, 2).

    Notes:
        Based on cs_total_variation from https://github.com/facebookresearch/fastMRI/blob/main/fastmri_examples/cs/run_bart.py
    """
    kspace = kspace.cpu().permute(1, 2, 0, 3).unsqueeze(0)
    kspace = fastmri.tensor_to_complex_np(kspace)
    print('Estimating sensitivity maps...', end='', flush=True)
    sens_maps = bart(1, f'ecalib -d2 -m1 {sens_params}', kspace)
    sens_maps = sens_maps.squeeze().transpose(2, 0, 1)
    return T.to_tensor(sens_maps)


def get_cached_sens_maps(kspace: torch.Tensor, file_name: str, slice_index: int, sens_params: str) -> torch.Tensor:
    """
    Load cached sensitivity maps. If no cached sensitivity maps with the given parameters are found, they are estimated
        based on the input kspace and then saved.

    Args:
        kspace: kspace of shape (num_coils, H, W, 2).
        file_name: File name.
        slice_index: Index of the slice for which to load the sensitivity maps.
        sens_params: Parameters that get directly passed to ecalib from the Bart toolbox. See get_sens_maps for details.

    Returns:
        The sensitivity maps of shape (num_coils, H, W, 2).
    """
    sens_file_path = paths.get_sens_file_path(file_name, slice_index, sens_params)

    if os.path.exists(sens_file_path):
        return torch.load(sens_file_path)
    else:
        os.makedirs(os.path.dirname(sens_file_path), exist_ok=True)
        sens_maps = get_sens_maps(kspace, sens_params)
        torch.save(sens_maps, sens_file_path)
        return sens_maps


def get_model(prior: str, device: torch.device) -> Glow:
    """
    Load a Glow model from a checkpoint. The parameters are read from the corresponding yaml file located in the config/ directory.
        If prior is one of 'including_fat' or 'excluding_fat', the checkpoint is automatically downloaded.

    Args:
        prior: The name of the model to load.
        device: The device to put the model on.

    Returns:
        A Glow model in eval mode.
    """
    params = get_prior_params(prior)
    model = Glow(width=params.width,
                 depth=params.depth,
                 n_levels=params.n_levels,
                 input_dims=params.input_dims,
                 affine_scale_low=params.affine_scale_low,
                 unet_coupling=params.unet_coupling,
                 try_to_center=params.try_to_center).to(device)

    checkpoint_path = os.path.join(paths.get_model_path(), f'{prior}_prior.pt')
    if not os.path.exists(checkpoint_path):
        download(url=get_prior_url(prior),
                 save_path=checkpoint_path)
    model_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_checkpoint['state_dict'])
    model.eval()
    return model


def get_prior_params(prior: str) -> dotdict:
    """Load the parameters of prior from the corresponding yaml file."""
    config_path = os.path.join(paths.get_config_path(), f'{prior}_prior.yaml')
    return load_config(config_path)


def load_config(config_path: str) -> dotdict:
    """Return the config saved in a yaml file at config_path."""
    with open(config_path) as f:
        return dotdict(yaml.full_load(f))


def download(url: str, save_path: str):
    """Download a file from url and save it at save_path."""
    print(f'Downloading {os.path.basename(save_path)}. This might take a while...')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with urlopen(url) as response, open(save_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f'Download successful. The file was saved at {save_path}')


def get_prior_url(prior: str) -> str:
    if prior == 'including_fat':
        return 'https://dl.dropboxusercontent.com/s/pcljd4hihq7gx1v/including_fat_prior.pt?dl=0'
    elif prior == 'excluding_fat':
        return 'https://dl.dropboxusercontent.com/s/6obm2no77j3jjga/excluding_fat_prior.pt?dl=0'
    else:
        raise ValueError(f'Prior {prior} is not available for download. '
                         f'You can train your own priors using train_glow.py')


def get_HW(x: Union[torch.Tensor, np.ndarray]) -> Tuple[int, int]:
    return x.shape[-3:-1]


def get_crop_size(crop_size: CropSize, kspace_shape: Tuple[int, ...]) -> Tuple[int, int]:
    if isinstance(crop_size, tuple):
        assert len(crop_size) == 2
    else:
        crop_size = (crop_size, crop_size)
    return tuple(size if size is not None else kspace_shape[-3 + dim] for dim, size in enumerate(crop_size))  # type: ignore


def get_device(local_rank: Optional[int]) -> torch.device:
    if local_rank is not None and torch.cuda.is_available():
        device = f'cuda:{local_rank}'
    else:
        device = 'cpu'
    return torch.device(device)


def get_seed(file_path: str) -> List[int]:
    file_name = os.path.basename(file_path)
    return list(map(ord, file_name))


def get_timestamp() -> str:
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())


def print_exception(e):
    print(''.join(TracebackException.from_exception(e).format()))


def to_tuple_of_int(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(num) for num in x)


class InstabilityError(ValueError):
    pass
