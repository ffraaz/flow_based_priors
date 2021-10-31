import unittest

import torch

import fastmri
from fastmri.data import transforms as T

from kernprior import paths
from kernprior import utils

from helpers import get_kspace, get_num_mid_slices


class TestUtils(unittest.TestCase):
    def test_load_kspace_and_target(self):
        for challenge in ['singlecoil', 'multicoil']:
            file_path = paths.get_data_file_path('val', challenge, 'file1000831.h5')
            kspace = utils.load_kspace_slice(file_path)
            target = utils.load_target(file_path)

            expected_kspace_n_dim = 4 if challenge == 'multicoil' else 3
            self.assertTrue(kspace.ndim == expected_kspace_n_dim)
            self.assertTrue(kspace.shape[-1] == 2)
            self.assertTrue(target.shape == (320, 320))

    def test_get_mid_slice_indices(self):
        for num_slices in [25, 30, 32]:
            for mid_slice_range in [0, 1, 2, 5]:
                mid_slice_indices = utils.get_mid_slice_indices(num_slices, mid_slice_range)
                self.assertTrue(len(mid_slice_indices) == get_num_mid_slices(mid_slice_range))

        with self.assertRaises(AssertionError):
            utils.get_mid_slice_indices(num_slices=5, mid_slice_range=4)

    def test_get_sens_maps(self):
        for sens_params in ['', '-W -c0']:
            kspace = get_kspace(challenge='multicoil')
            sens_maps = utils.get_sens_maps(kspace, sens_params)
            sens_maps_intensity = fastmri.rss_complex(sens_maps)
            sens_maps_intensity_center = T.center_crop(sens_maps_intensity, (100, 100))

            self.assertTrue(sens_maps.shape == kspace.shape)
            self.assertTrue(torch.logical_or(sens_maps_intensity == 0, sens_maps_intensity.isclose(torch.tensor(1.))).all())
            self.assertTrue(torch.allclose(sens_maps_intensity_center, torch.tensor(1.)))

    def test_get_crop_size(self):
        kspace_shape = get_kspace(challenge='multicoil').shape
        self.assertTrue(utils.get_crop_size(None, kspace_shape) == (640, 368))
        self.assertTrue(utils.get_crop_size((None, None), kspace_shape) == (640, 368))
        self.assertTrue(utils.get_crop_size((320, 320), kspace_shape) == (320, 320))
        self.assertTrue(utils.get_crop_size((None, 300), kspace_shape) == (640, 300))
        self.assertTrue(utils.get_crop_size((250, None), kspace_shape) == (250, 368))
        self.assertTrue(utils.get_crop_size(270, kspace_shape) == (270, 270))

    def test_get_HW(self):
        self.assertTrue(utils.get_HW(torch.zeros(300, 200, 2)) == (300, 200))
        self.assertTrue(utils.get_HW(torch.zeros(15, 450, 100, 2)) == (450, 100))


if __name__ == '__main__':
    unittest.main()
