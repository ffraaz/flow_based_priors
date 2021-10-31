import numpy as np
import unittest

import torch

import fastmri
from fastmri import complex_abs, ifft2c
from fastmri.data import transforms as T
from fastmri.data.subsample import create_mask_for_mask_type

from kernprior import transforms
from kernprior import utils
from kernprior.reconstruct import init_zs_to_zero

from helpers import get_kspace


class TestTransforms(unittest.TestCase):
    def test_to_from_chan_complex(self):
        shapes = [
            (1, 6, 40, 30),
            (5, 2, 50, 60),
            (32, 4, 10, 10),
            (3, 2, 320, 320)
        ]

        for shape in shapes:
            x = torch.randn(shape)
            x_complex = transforms.chan_to_complex(x)
            x_chan = transforms.complex_to_chan(x_complex)
            self.assertTrue(x_complex.ndim == x.ndim + 1)
            self.assertTrue(x_complex.shape[-1] == 2)
            self.assertTrue((x_chan == x).all())

    def test_polar_cartesian(self):
        x = get_kspace(challenge='multicoil')
        x_polar = transforms.to_polar(x)
        x_cartesian = transforms.to_cartesian(x_polar)
        self.assertTrue(torch.allclose(x_cartesian, x))
        self.assertTrue(torch.allclose(x_polar[...,0], complex_abs(x)))

    def test_generate(self):
        height = 640
        width = 368
        overlap = 0
        device = utils.get_device(local_rank=None)

        for prior in ['including_fat', 'excluding_fat']:
            glow = utils.get_model(prior, device)
            input_dims = utils.get_prior_params(prior).input_dims
            patch_grid_shape, *_ = transforms.get_compose_params(height, width, input_dims[-2:], overlap)
            n_patches = np.prod(patch_grid_shape)
            zs = init_zs_to_zero(glow, input_dims, n_patches, device)

            patches, _ = glow.inverse(zs)
            x = transforms.generate(zs, glow, height, width, overlap)

            self.assertTrue(patches.shape == (n_patches, *input_dims))
            self.assertTrue(x.ndim == 3)
            self.assertTrue(utils.get_HW(x) == (height, width))
            self.assertTrue(x.shape[-1] == 2)

    def test_compose(self):
        test_configs = [
            (320, 320, 64, 0, 2),
            (257, 257, 64, 0, 2),
            (320, 320, 64, 10, 2),
            (320, 320, 64, 0, 1),
            (100, 111, (30, 25), 5, 3),
            (120, 90, (28, 23), (4, 6), 5),
            (50, 30, 20, (3, 4), 2),
            (1, 2, 1, 0, 3),
            (1, 1, 3, 0, 4),
        ]

        test_configs_invalid = [
            (0, 0, 5, 0, 2),
            (320, 320, 64, 65, 2),
            (1, 1, 3, 1, 4),
        ]

        for height, width, patch_size, overlap, n_channels in test_configs:
            grid_shape, output_size, patch_size, stride = transforms.get_compose_params(height, width, patch_size, overlap)
            input_shape = (np.prod(grid_shape), n_channels, *patch_size)
            x = torch.ones(input_shape)  # type: ignore
            x_folded = transforms.compose(x, height, width, overlap)

            self.assertTrue((x_folded == 1).all())
            self.assertTrue(x_folded.shape == (n_channels, height, width))

            if overlap == 0:
                self.assertTrue((transforms.compose_additive(x, height, width, overlap) == 1).all())

        for height, width, patch_size, overlap, n_channels in test_configs_invalid:
            with self.assertRaises(AssertionError):
                transforms.get_compose_params(height, width, patch_size, overlap)

    def test_crop_spatially(self):
        kspace = get_kspace(challenge='multicoil')
        crop_size = (300, 310)
        kspace_cropped = transforms.crop_spatially(kspace, crop_size=crop_size)
        self.assertTrue(utils.get_HW(kspace_cropped) == crop_size)

    def test_get_image(self):
        for challenge in ['singlecoil', 'multicoil']:
            kspace = get_kspace(challenge)
            image = transforms.get_image(kspace, challenge)
            self.assertTrue(image.shape == utils.get_HW(kspace))

    def test_forward_model(self):
        for challenge in ['multicoil', 'singlecoil']:
            kspace = get_kspace(challenge)
            mask_func = create_mask_for_mask_type('random',
                                                  accelerations=[4],
                                                  center_fractions=[0.08])

            masked_kspace, mask = T.apply_mask(kspace, mask_func)
            x = torch.ones(kspace.shape[-3:])

            forward_model = transforms.get_forward_model(masked_kspace, mask, challenge, sens_params='')
            y_hat = forward_model(x)

            self.assertTrue(y_hat.shape == kspace.shape)
            self.assertTrue(y_hat[~mask.expand(y_hat.shape).bool()].count_nonzero() == 0)

    def test_data_consistency(self):
        height = 640
        width = 368
        x = torch.zeros(height, width, 2)
        kspace = torch.ones(15, height, width, 2)

        mask_func = create_mask_for_mask_type('random',
                                              accelerations=[4],
                                              center_fractions=[0.08])

        masked_kspace, mask = T.apply_mask(kspace, mask_func)

        x_dc = transforms.data_consistency(x, masked_kspace, mask, sens_params='', challenge='multicoil')
        self.assertTrue(x_dc.shape == utils.get_HW(masked_kspace))

    def test_sens_reduce(self):
        kspace = get_kspace(challenge='multicoil')
        image = ifft2c(kspace)
        sens_maps = utils.get_sens_maps(kspace)
        x_reduced = transforms.sens_reduce(image, sens_maps)
        x_expanded = transforms.sens_expand(x_reduced, sens_maps)
        x_expanded_reduced = transforms.sens_reduce(x_expanded, sens_maps)
        self.assertTrue(x_reduced.ndim == 3)
        self.assertTrue(x_reduced.shape == image.shape[-3:])
        self.assertTrue(x_expanded.shape == image.shape)
        self.assertTrue(torch.allclose(x_reduced, x_expanded_reduced))

    def test_sens_reduce_precise(self):
        kspace = get_kspace(challenge='multicoil')
        image = ifft2c(kspace)
        sens_maps = utils.get_sens_maps(kspace)
        x_rss = fastmri.rss_complex(image)
        x_reduced = transforms.sens_reduce(image, sens_maps)
        x_reduced_precise = transforms.sens_reduce_precise(image, sens_maps)
        self.assertTrue(torch.allclose(complex_abs(x_reduced_precise), x_rss))
        self.assertTrue(torch.allclose(transforms.to_polar(x_reduced_precise)[...,1], transforms.to_polar(x_reduced)[...,1]))


if __name__ == '__main__':
    unittest.main()
