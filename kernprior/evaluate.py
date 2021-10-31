import os
from argparse import ArgumentParser
from copy import deepcopy
import torch

from fastmri import complex_abs
from fastmri.data import transforms as T

from kernprior import metrics
from kernprior import paths
from kernprior import transforms
from kernprior import utils
from kernprior.data import get_mri_dataset_eval, get_stanford_dataset_eval
from kernprior.reconstruct import reconstruct


def run(common_args):
    common_args.device = utils.get_device(common_args.local_rank)

    if common_args.dataset_name == 'stanford':
        dataset = get_stanford_dataset_eval(common_args)
    else:
        dataset = get_mri_dataset_eval(common_args)

    n_samples = len(dataset)
    assert n_samples == get_expected_n_samples(common_args.dataset_name, common_args.acquisition)

    print(f'device: {common_args.device}')
    print(f'Started evaluation. Tensorboard logs are saved to {paths.get_evaluation_path(common_args.evaluation_name)}')

    summary_file = os.path.join(paths.get_evaluation_path(common_args.evaluation_name), 'summary.csv')

    for i in range(n_samples):
        masked_kspace, mask, target, file_name, slice_num, acquisition = dataset[i]

        args = deepcopy(common_args)

        args.prior = acquisition if acquisition == 'including_fat' else 'excluding_fat'
        args.lr = args.lr[args.prior]
        args.gamma = args.gamma[args.prior]
        args.file_name = file_name

        if args.dataset_name != 'stanford':
            args.file_path = paths.get_data_file_path(args.dataset, args.challenge, file_name, args.anatomy)

        args.save_dir = paths.get_evaluation_path(args.evaluation_name, file_name, slice_num)

        print(f'Started reconstruction {i+1} / {n_samples} ({file_name})')
        try:
            x, x_abs, x_dc = reconstruct(masked_kspace, mask, target, args)
            save_results(x_abs, x_dc, target, file_name, slice_num, summary_file, True, acquisition, args.evaluation_name)
            print(f'Done with reconstruction of {file_name}')
        except utils.InstabilityError:
            print(f'Stopped reconstruction of {file_name} early due to unstable inverse pass')
            x_abs, x_dc = recover(masked_kspace, mask, args)
            save_results(x_abs, x_dc, target, file_name, slice_num, summary_file, False, acquisition, args.evaluation_name)
        except Exception as e:
            utils.print_exception(e)

        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
        print()


def recover(masked_kspace, mask, args):
    zs_path = os.path.join(args.save_dir, 'zs.pt')
    if os.path.exists(zs_path):
        zs = torch.load(zs_path)
        x_abs, x_dc = restore_from_checkpoint(zs, masked_kspace, mask, args)
        print('Included results from an earlier checkpoint')
    else:
        x_abs = transforms.get_image(masked_kspace, challenge=args.challenge)
        x_dc = x_abs
        print('Could not find an earlier checkpoint. Instability occurred really early. Used zero filled reconstruction instead')

    torch.save(x_dc, os.path.join(args.save_dir, 'x_dc.pt'))
    return x_abs, x_dc


def restore_from_checkpoint(zs, masked_kspace, mask, args):
    glow = utils.get_model(args.prior, args.device)
    height, width = utils.get_HW(masked_kspace)

    with torch.no_grad():
        x = transforms.generate(zs, glow, height, width, args.overlap)

    x_abs = complex_abs(x)
    x_dc = transforms.data_consistency(x, masked_kspace.to(x.device), mask.to(x.device), args.sens_params, args.challenge)

    return x_abs, x_dc


def save_results(x_abs, x_dc, target, file_name, slice_num, summary_file, stable, acquisition, evaluation_name):
    x_abs = T.center_crop(x_abs, target.shape)
    x_dc = T.center_crop(x_dc, target.shape)

    res_line = [evaluation_name,
                file_name,
                int(slice_num),
                metrics.ssim(target, x_abs, normalize_to='target'),
                metrics.ssim(target, x_dc, normalize_to='target'),
                metrics.ssim(target, x_dc, normalize_to='pred'),
                metrics.vif(target, x_dc, normalize_to='target'),
                metrics.vif(target, x_dc, normalize_to='pred'),
                metrics.psnr(target, x_dc, normalize_to='target'),
                metrics.psnr(target, x_dc, normalize_to='pred'),
                acquisition,
                stable]

    with open(summary_file, 'a') as f:
        print(','.join(map(str, res_line)), file=f)


def get_expected_n_samples(dataset_name, acquisition):
    if dataset_name == 'knee_val':
        if acquisition is None:
            return 199
        elif acquisition == 'including_fat':
            return 100
        elif acquisition == 'excluding_fat':
            return 99
    elif dataset_name == 'stanford':
        return 18
    elif dataset_name == 'fastmri_a':
        return 105
    elif dataset_name == 'brain_100':
        return 100


def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--dataset_name',
        choices=('knee_val', 'brain_100', 'stanford', 'fastmri_a'),
        default='stanford',
        type=str,
        help='Which dataset to evaluate on'
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        help='Which GPU to use'
    )

    config_path = os.path.join(paths.get_config_path(), 'reconstruction.yaml')
    args = utils.load_config(config_path)

    args.update(parser.parse_args().__dict__)

    if args.dataset_name != 'stanford':
        args.anatomy = 'brain' if args.dataset_name == 'brain_100' else 'knee'

    args.evaluation_name = f'{args.dataset_name}_{utils.get_timestamp()}'

    args.pad_sides = args.dataset_name == 'brain_100'

    run(args)


if __name__ == '__main__':
    main()
