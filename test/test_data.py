from itertools import product
import os
import unittest

from fastmri import complex_abs

from kernprior import data
from kernprior import paths
from kernprior import transforms
from kernprior import utils

from helpers import get_num_mid_slices


class TestData(unittest.TestCase):
    def test_FastMRIDataset(self):
        for acquisition, adversarially_filtered in product((None, 'including_fat', 'excluding_fat'),
                                                           [False, True]):
            dataset = data.FastMRIDataset(root=paths.get_mri_data_path('val', 'multicoil'),
                                          challenge='multicoil',
                                          transform=None,
                                          mid_slice_range=4 if adversarially_filtered else 0,
                                          acquisition=acquisition,
                                          adversarially_filtered=adversarially_filtered)

            self.assertTrue(len(dataset) > 0)
            self.assertTrue(dataset[0] is not None)
            self.assertTrue(dataset[-1] is not None)

        for mid_slice_range in range(5):
            data_path = paths.get_mri_data_path('val', 'multicoil')
            dataset = data.FastMRIDataset(root=data_path,
                                          challenge='multicoil',
                                          transform=None,
                                          mid_slice_range=mid_slice_range)

            expected_dataset_size = get_num_mid_slices(mid_slice_range) * len(os.listdir(data_path))
            self.assertTrue(len(dataset) == expected_dataset_size)

    def test_StanfordDataset(self):
        data_transform = data.StanfordDataTransform(acceleration=4, center_fraction=0.08, normalize=True)
        dataset = data.StanfordDataset(root=paths.get_stanford_data_path(), transform=data_transform, download=True)

        self.assertTrue(len(dataset) == 18)
        self.assertTrue(dataset[0] is not None)
        self.assertTrue(dataset[-1] is not None)

    def test_TrainDataTransform(self):
        for patch_size, challenge, sens_params, reduce_precise in product([32, 64, 128],
                                                                          ['multicoil', 'singlecoil'],
                                                                          ['', '-W -c0'],
                                                                          [True, False]):
            data_transform = data.TrainDataTransform(patch_size, challenge, sens_params, reduce_precise)
            dataset = data.FastMRIDataset(root=paths.get_mri_data_path('train', challenge),
                                          challenge=challenge,
                                          transform=data_transform,
                                          mid_slice_range=0)

            patch, _ = dataset[0]
            self.assertTrue(patch.shape == (2, patch_size, patch_size))
            self.assertTrue((complex_abs(transforms.chan_to_complex(patch.unsqueeze(0))) <= 1).all())

    def test_InferenceDataTransform(self):
        for challenge in ['multicoil', 'singlecoil']:
            dataset = data.FastMRIDataset(root=paths.get_mri_data_path('val', challenge),
                                          challenge=challenge,
                                          mid_slice_range=0)

            for mask_type, acceleration, crop_size, normalize, use_seed in product(['random', 'equispaced'],
                                                                                   [4, 8],
                                                                                   [None, (320, 320), (570, None)],
                                                                                   [True, False],
                                                                                   [True, False]):

                data_transform = data.InferenceDataTransform(mask_type=mask_type,
                                                             acceleration=acceleration,
                                                             center_fraction=0.08 if acceleration == 4 else 0.04,
                                                             challenge=challenge,
                                                             crop_size=crop_size,
                                                             normalize=normalize,
                                                             pad_sides=False,
                                                             use_seed=use_seed)

                masked_kspace, mask, _, file_name, *_ = data_transform(*dataset[0])
                file_path = paths.get_data_file_path('val', challenge, file_name)

                self.assertTrue(mask.count_nonzero() <= mask.numel() / acceleration * 1.5)  # a bit of slack since RandomMaskFunc doesn't provide guarantees

                expected_n_dim = 4 if challenge == 'multicoil' else 3
                self.assertTrue(masked_kspace.ndim == mask.ndim == expected_n_dim)

                self.assertTrue(masked_kspace.shape[-1] == 2)
                self.assertTrue(masked_kspace.shape[-2] == mask.shape[-2])
                self.assertTrue(mask.shape[-2] == mask.numel())

                if crop_size:
                    expected_spatial_size = utils.get_crop_size(crop_size, utils.load_kspace_slice(file_path).shape)
                    self.assertTrue(utils.get_HW(masked_kspace) == expected_spatial_size)

                if normalize:
                    image_max = transforms.get_image(masked_kspace, challenge).max()
                    self.assertTrue(round(float(image_max), ndigits=1) == 1)

                if use_seed:
                    _, second_mask, *_ = data_transform(*dataset[0])
                    self.assertTrue((mask == second_mask).all())  # type: ignore


if __name__ == '__main__':
    unittest.main()
