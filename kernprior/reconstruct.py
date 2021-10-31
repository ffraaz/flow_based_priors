import numpy as np
import os
from argparse import ArgumentParser
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from fastmri import complex_abs
from fastmri.data import transforms as T

from kernprior import data
from kernprior import metrics
from kernprior import paths
from kernprior import transforms
from kernprior import utils


def reconstruct(
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    target: Union[np.ndarray, torch.Tensor],
    args
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    writer = SummaryWriter(log_dir=args.save_dir)
    torch.save(dict(args), os.path.join(args.save_dir, 'args.pt'))

    height, width = utils.get_HW(masked_kspace)
    input_dims = utils.get_prior_params(args.prior).input_dims
    patch_grid_shape, *_ = transforms.get_compose_params(height, width, input_dims[-2:], args.overlap)
    n_patches = np.prod(patch_grid_shape)

    glow = utils.get_model(args.prior, args.device)
    zs = init_zs_to_zero(glow, input_dims, n_patches, args.device)
    optimizer = torch.optim.Adam(zs, lr=args.lr)

    mask = mask.to(args.device)
    masked_kspace = masked_kspace.to(args.device)

    forward_model = transforms.get_forward_model(masked_kspace, mask, args.challenge, args.sens_params)

    for i in range(args.n_iter):
        x = transforms.generate(zs, glow, height, width, args.overlap)

        if is_unstable(x):
            save_and_stop(zs, x, args.save_dir)

        y_hat = forward_model(x)

        loss_data = F.mse_loss(y_hat, masked_kspace, reduction='sum')
        loss_prior = sum(z.norm().square() for z in zs)
        loss = loss_data + args.gamma * loss_prior

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_and_log(zs, x, loss_data, loss_prior, loss, writer, i, target, args)

    with torch.no_grad():
        x = transforms.generate(zs, glow, height, width, args.overlap)

    x_abs = complex_abs(x)
    x_dc = transforms.data_consistency(x, masked_kspace, mask, args.sens_params, args.challenge)

    torch.save(x, os.path.join(args.save_dir, 'x.pt'))
    torch.save(x_abs, os.path.join(args.save_dir, 'x_abs.pt'))
    torch.save(x_dc, os.path.join(args.save_dir, 'x_dc.pt'))

    return x, x_abs, x_dc


def init_zs_to_zero(glow, input_dims, batch_size, device):
    # perform one forward pass in order to know the shape of the latent vector
    example_input = torch.zeros(batch_size, *input_dims).to(device)
    with torch.no_grad():
        zs, _ = glow(example_input)
    return [torch.zeros_like(z, requires_grad=True) for z in zs]


def is_unstable(x):
    return x.isnan().any() or x.gt(100).any()


def save_and_stop(zs, x, save_dir):
    print(x.max())
    torch.save(zs, os.path.join(save_dir, 'zs_unstable.pt'))
    torch.save(x, os.path.join(save_dir, 'x_unstable.pt'))
    raise utils.InstabilityError('Unstable inverse pass')


def save_and_log(zs, x, loss_data, loss_prior, loss, writer, i, target, args):
    x = x.detach()
    x = complex_abs(x)
    x_center = T.center_crop(x, target.shape)

    if i % args.log_interval == 0:
        writer.add_scalar('F.mse_loss(y_hat, masked_kspace)', loss_data.item(), i)
        writer.add_scalar('loss_prior', loss_prior.item(), i)
        writer.add_scalar('loss_data + gamma * loss_prior', loss.item(), i)
        writer.add_scalar('ssim(target, x)', metrics.ssim(target, x_center, normalize_to='target'), i)

    if i % args.save_interval == 0 or i == args.n_iter - 1:
        x_center = transforms.normalize_image(x_center)
        writer.add_image('x', x, global_step=i, dataformats='HW')
        writer.add_image('x_center', x_center, global_step=i, dataformats='HW')
        torch.save(zs, os.path.join(args.save_dir, 'zs.pt'))


def run(args):
    args.file_path = paths.get_data_file_path(args.dataset, args.challenge, args.file_name)

    args.prior = utils.get_acquisition(args.file_path)
    args.lr = args.lr[args.prior]
    args.crop_size = args.crop_size[args.prior]
    args.gamma = args.gamma[args.prior]

    args.device = utils.get_device(args.local_rank)
    args.save_dir = paths.get_save_dir(args.file_name)

    target = utils.load_target(args.file_path, challenge=args.challenge)  # only used for logging
    kspace = utils.load_kspace_slice(args.file_path, as_tensor=False)  # conversion to torch.Tensor is done in data_transform

    data_transform = data.InferenceDataTransform(mask_type='random',
                                                 acceleration=args.acceleration,
                                                 center_fraction=args.center_fraction,
                                                 challenge=args.challenge,
                                                 crop_size=args.crop_size,
                                                 normalize=args.normalize,
                                                 pad_sides=args.pad_sides)

    masked_kspace, mask, *_ = data_transform(kspace,
                                             attrs=utils.get_attrs(args.file_path),
                                             file_name=os.path.basename(args.file_path))

    print(f'device: {args.device}')
    print(f'Starting reconstruction of {args.file_name}. Tensorboard logs are saved to {args.save_dir}')
    reconstruct(masked_kspace, mask, target, args)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--file_name',
        default='file1000831.h5',
        type=str,
        help='Which image to reconstruct'
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        help='Which GPU to use'
    )

    config_path = os.path.join(paths.get_config_path(), 'reconstruction.yaml')
    args = utils.load_config(config_path)

    args.update(parser.parse_args().__dict__)

    run(args)


if __name__ == '__main__':
    main()
