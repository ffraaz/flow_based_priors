import numpy as np
import os
import pickle
from typing import Dict, Optional, Tuple, Union, Mapping, Callable, Set

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop

from fastmri import complex_abs, ifft2c
from fastmri.data import SliceDataset
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc, create_mask_for_mask_type

from kernprior import paths
from kernprior import transforms
from kernprior import utils


class FastMRIDataset(SliceDataset):
    """
    fastMRI Dataset (https://fastmri.org/dataset/).
    """

    def __init__(
        self,
        mid_slice_range: Optional[int] = None,
        acquisition: Optional[str] = None,
        subset: Optional[set] = None,
        adversarially_filtered: bool = False,
        **kwargs,
    ):
        """
        Args:
            mid_slice_range: Optional; If provided, only add the middle slice +/- mid_slice_range
                of each volume to the dataset.
            acquisition: Optional; If provided, only include examples from this acquisition type
                ('including_fat' or 'excluding_fat' for knees) in the dataset.
            subset: Optional; Only files from subset are included.
            adversarially_filtered: Optional; Whether to only include files from the fastMRI-A dataset
                (see https://arxiv.org/abs/2102.06103).
            kwargs: Keyword arguments are passed to fastmri.data.SliceDataset.
        """
        super().__init__(**kwargs)

        if mid_slice_range is not None:
            # each example is a tuple with 3 elements: (file_path, slice_num, metadata)
            file_path_to_mid_slice_indices = self._get_file_path_to_mid_slice_indices(mid_slice_range)
            self.examples = [ex for ex in self.examples if ex[1] in file_path_to_mid_slice_indices[ex[0]]]

        if acquisition:
            file_path_to_acquisition = self._get_file_path_to_acquisition()
            self.examples = [ex for ex in self.examples if file_path_to_acquisition[ex[0]] == acquisition]

        if subset:
            self.examples = [ex for ex in self.examples if ex[0].name in subset]

        if adversarially_filtered:
            fastmri_a_dataset = get_fastmri_a_dataset()
            self.examples = [
                ex
                for ex in self.examples
                if (ex[0].name, ex[1]) in fastmri_a_dataset
            ]

    def _get_file_path_to_mid_slice_indices(self, mid_slice_range: int):
        file_path_to_mid_slice_indices = {}
        for file_path, *_ in self.examples:
            if file_path not in file_path_to_mid_slice_indices:
                num_slices = utils.get_num_slices(file_path)
                file_path_to_mid_slice_indices[file_path] = set(utils.get_mid_slice_indices(num_slices, mid_slice_range))
        return file_path_to_mid_slice_indices

    def _get_file_path_to_acquisition(self):
        file_path_to_acquisition = {}
        for file_path, *_ in self.examples:
            if file_path not in file_path_to_acquisition:
                file_path_to_acquisition[file_path] = utils.get_acquisition(file_path)
        return file_path_to_acquisition


class StanfordDataset(Dataset):
    """
    Stanford Dataset (collected by Darestani et al. https://arxiv.org/abs/2102.06103).
    """

    def __init__(self, root: str, transform: Optional[Callable] = None, download: bool = False):
        """
        Args:
            root: Path to the dataset.
            transform: Optional; A callable object that pre-processes the data. The transform should take
                'file_name', 'kspace', 'target' as inputs.
            download: Optional; If True, downloads the dataset and saves it at root.
                If the dataset is already downloaded, it is not downloaded again.
        """
        self.root = root
        self.transform = transform
        self.examples = []

        if not os.path.exists(self.root):
            if download:
                utils.download(url='https://rice.box.com/shared/static/4xk6nef26vk8uyes4wymtob5pbmcfdyd',
                               save_path=self.root)
            else:
                raise RuntimeError('Dataset not found. Use download=True to download.')

        with open(self.root, 'rb') as f:
            dataset = pickle.load(f)

        for file_name, sample in dataset.items():
            self.examples.append((file_name, sample['kspace'], sample['reconstruction_rss']))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        sample = self.examples[i]

        if self.transform is not None:
            sample = self.transform(*sample)

        return sample


class TrainDataTransform:
    """
    Data transform for training a flow-based prior on patches of MRI data. Used as 'transform' for FastMRIDataset.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int], Tuple[int, int]],
        challenge: str,
        sens_params: str,
        reduce_precise: bool,
    ):
        """
        Args:
            patch_size: The patch size of the prior.
            challenge: One of ('singlecoil', 'multicoil').
            sens_params: Parameters that get directly passed to ecalib from the Bart toolbox.
                See utils.get_sens_maps for details.
            reduce_precise: Whether to replace the magnitude of the MVUE image with the RSS estimate. Only relevant
                if challenge == 'multicoil'.
        """
        self.random_crop = RandomCrop(patch_size)
        self.challenge = challenge
        self.sens_params = sens_params
        self.center_crop_size = (320, 320)
        self.reduce_precise = reduce_precise

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        file_name: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            kspace: kspace of shape (num_coils, H, W) for multi-coil data or (H, W) for single-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            file_name: File name.
            slice_num: Index of the slice.

        Returns:
            tuple containing:
                patch: Complex-valued patch of shape (2, patch_H, patch_W), i.e. real and imaginary dims are stacked as
                    channel dim.
                slice_num: Index of the slice.

        Notes:
            Docstring based on https://github.com/facebookresearch/fastMRI/blob/master/fastmri/data/transforms.py
        """
        kspace = T.to_tensor(kspace)
        target = ifft2c(kspace)
        target = T.complex_center_crop(target, self.center_crop_size)

        if self.challenge == 'multicoil':
            # combine individual coil images into a single complex-valued image
            sens_maps = utils.get_cached_sens_maps(kspace, file_name, slice_num, self.sens_params)
            sens_maps = T.complex_center_crop(sens_maps, self.center_crop_size)
            if self.reduce_precise:
                target = transforms.sens_reduce_precise(target, sens_maps)
            else:
                target = transforms.sens_reduce(target, sens_maps)

        target = target / complex_abs(target).max()  # normalize
        target = target.permute(2, 0, 1)  # complex dim as channel dim
        patch = self.random_crop(target)

        return patch, slice_num


class InferenceDataTransform:
    """
    Data transform for obtaining undersampled kspace slices.
    """

    def __init__(
        self,
        mask_type: str,
        acceleration: int,
        center_fraction: float,
        challenge: str,
        crop_size: Union[utils.CropSize, Mapping[str, utils.CropSize]],
        normalize: bool,
        pad_sides: bool = False,
        use_seed: bool = True,
    ):
        """
        Args:
            mask_type: One of ('random', 'equispaced').
            acceleration: The desired undersampling factor.
            center_fraction: Fraction of low-frequency columns to be retained.
            challenge: One of ('singlecoil', 'multicoil').
            crop_size: The kspace is spatially cropped to crop_size before undersampling. If a dimension of crop_size is
                None, the kspace does not get cropped in this dimension. If crop_size is a Dict, it should map from
                'including_fat' and 'excluding_fat' to a crop_size to use for the respective acquisition type.
            normalize: Whether to divide the kspace by the maximum pixel value of the zero-filled reconstruction.
            pad_sides: Optional; Whether to pad the sides of the kspace with zeros.
            use_seed: Optional; Whether to seed the random number generator used for creating the mask based on the file_name.
        """
        self.acceleration = acceleration
        self.challenge = challenge
        self.crop_size = crop_size
        self.normalize = normalize
        self.pad_sides = pad_sides
        self.use_seed = use_seed
        self.mask_func = create_mask_for_mask_type(mask_type,
                                                   accelerations=[acceleration],
                                                   center_fractions=[center_fraction])

    def __call__(
        self,
        kspace: np.ndarray,
        mask: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        attrs: Optional[Dict] = None,
        file_name: Optional[str] = None,
        slice_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[str], Optional[int], Optional[str]]:
        """
        Args:
            kspace: kspace of shape (num_coils, H, W) for multi-coil data or (H, W) for single-coil data.
            mask: Optional; Mask from the test dataset.
            target: Optional; Target image.
            attrs: Optional; Acquisition related information stored in the HDF5 object.
            file_name: Optional; File name.
            slice_num: Optional; Index of the slice.

        Returns:
            tuple containing:
                masked_kspace: Undersampled kspace.
                mask: Undersampling mask used to obtain masked_kspace.
                target: Target image.
                file_name: File name.
                slice_num: Index of the slice.
                acquisition: The acquisition type of the measurement.

        Notes:
            Docstring based on https://github.com/facebookresearch/fastMRI/blob/master/fastmri/data/transforms.py
        """
        kspace = T.to_tensor(kspace)

        if target is not None:
            target = T.to_tensor(target)

        kspace_shape_uncropped = kspace.shape
        acquisition = utils.from_confusing_terminology(attrs['acquisition']) if attrs else None

        if isinstance(self.crop_size, dict):
            crop_size = self.crop_size[acquisition] if acquisition == 'including_fat' else self.crop_size['excluding_fat']
        else:
            crop_size = self.crop_size

        if crop_size is not None:
            crop_size = utils.get_crop_size(crop_size, kspace.shape)
            kspace = transforms.crop_spatially(kspace, crop_size)

        if self.pad_sides and kspace.shape[-2] == kspace_shape_uncropped[-2]:  # only pad if width dimension was not cropped
            padding = (attrs['padding_left'], attrs['padding_right'])
        else:
            padding = None

        seed = utils.get_seed(file_name) if self.use_seed else None
        masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed, padding)

        if self.normalize:
            masked_kspace = transforms.normalize_kspace(masked_kspace, self.challenge)

        return masked_kspace, mask, target, file_name, slice_num, acquisition


class StanfordDataTransform:
    """
    Data transform for obtaining undersampled kspace slices from the StanfordDataset.
    """

    def __init__(self, acceleration: int, center_fraction: float, normalize: bool, use_seed: bool = True):
        """
        Args:
            acceleration: The desired undersampling factor.
            center_fraction: Fraction of low-frequency columns to be retained.
            normalize: Whether to divide the kspace by the maximum pixel value of the zero-filled reconstruction.
            use_seed: Optional; Whether to seed the random number generator used for creating the mask based on the file_name.
        """
        self.acceleration = acceleration
        self.normalize = normalize
        self.use_seed = use_seed
        self.mask_func = RandomMaskFunc(accelerations=[acceleration], center_fractions=[center_fraction])

    def __call__(
        self,
        file_name: str,
        kspace: np.ndarray,
        target: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, str]:
        """
        Args:
            file_name: File name.
            kspace: kspace of shape (num_coils, H, W).
            target: Target image.

        Returns:
            tuple containing:
                masked_kspace: Undersampled kspace.
                mask: Undersampling mask used to obtain masked_kspace.
                target: Target image.
                file_name: File name.
                slice_num: Index of the slice (-1; included for compatibility with the fastMRI dataset).
                acquisition: Indicates that a sample from the Stanford dataset is returned.
        """
        kspace = T.to_tensor(kspace)
        target = T.to_tensor(target)

        kspace = kspace.rot90(1, dims=(-3, -2))
        target = target.rot90(1, dims=(0, 1))

        seed = utils.get_seed(file_name) if self.use_seed else None
        masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)

        if self.normalize:
            masked_kspace = transforms.normalize_kspace(masked_kspace, 'multicoil')

        return masked_kspace, mask, target, file_name, -1, 'stanford'


def get_dataloader_train(args, data_dependent_init=False):
    """Based on https://github.com/kamenbliznashki/normalizing_flows"""
    dataset = get_mri_dataset_train(args)
    batch_size = args.batch_size_init if data_dependent_init else args.batch_size
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.device.type == 'cuda' else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)


def get_mri_dataset_train(args):
    return FastMRIDataset(root=paths.get_mri_data_path(args.dataset, args.challenge),
                          challenge=args.challenge,
                          transform=TrainDataTransform(args.patch_size, args.challenge, args.sens_params, args.reduce_precise),
                          mid_slice_range=args.mid_slice_range,
                          acquisition=args.acquisition)


def get_mri_dataset_eval(args):
    data_transform = InferenceDataTransform(mask_type='equispaced' if args.dataset_name == 'brain_100' else 'random',
                                            acceleration=args.acceleration,
                                            center_fraction=args.center_fraction,
                                            challenge=args.challenge,
                                            crop_size=args.crop_size,
                                            normalize=args.normalize,
                                            pad_sides=args.pad_sides)

    return FastMRIDataset(root=paths.get_mri_data_path(args.dataset, args.challenge, args.anatomy),
                          challenge=args.challenge,
                          transform=data_transform,
                          mid_slice_range=4 if args.dataset_name == 'fastmri_a' else 0,
                          acquisition=args.acquisition,
                          subset=get_brain_100_dataset() if args.dataset_name == 'brain_100' else None,
                          adversarially_filtered=args.dataset_name == 'fastmri_a')


def get_stanford_dataset_eval(args):
    data_transform = StanfordDataTransform(acceleration=args.acceleration,
                                           center_fraction=args.center_fraction,
                                           normalize=args.normalize)

    return StanfordDataset(root=paths.get_stanford_data_path(),
                           transform=data_transform,
                           download=True)


def get_fastmri_a_dataset() -> Set[Tuple[str, int]]:
    """
    Return the 105 (file_name, slice_num) combinations that constitute
        the fastMRI-A dataset (see https://arxiv.org/abs/2102.06103).
    """
    fastmri_a_dataset = set()
    with open(paths.get_fastmri_a_path(), 'r') as f:
        for line in f:
            file_name, slice_num = line.strip().split(',')
            fastmri_a_dataset.add((file_name, int(slice_num)))
    return fastmri_a_dataset


def get_brain_100_dataset() -> Set[str]:
    """Return the 100 names of files from the fastMRI brain validation set used for the anatomy shift experiment"""
    with open(paths.get_brain_100_path(), 'r') as f:
        return set(line.strip() for line in f)
