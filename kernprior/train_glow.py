"""Implementation taken from https://github.com/kamenbliznashki/normalizing_flows and modified slightly."""
import math
import os
import pprint
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import fastmri

from kernprior import paths
from kernprior import transforms
from kernprior import utils
from kernprior.data import get_dataloader_train, get_mri_dataset_train
from kernprior.glow import Glow


best_eval_logprob = float('-inf')


def train(model, train_dataloader, optimizer, writer, latent_noise, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train_epoch(model, train_dataloader, optimizer, writer, epoch, latent_noise, args)

    # save final model if it wasn't saved already
    if args.save_checkpoints and args.step % args.save_interval != 0:
        epoch = args.start_epoch + args.n_epochs - 1
        save_progress(model, optimizer, writer, epoch, args)


def train_epoch(model, dataloader, optimizer, writer, epoch, latent_noise, args):
    model.train()

    tic = time.time()
    data_loading_timer = time.perf_counter()
    for i, (x, y) in enumerate(dataloader):
        writer.add_scalar('batch_loading_time', time.perf_counter() - data_loading_timer, args.step)
        args.step += 1
        # warmup learning rate
        if epoch <= args.n_epochs_warmup:
            optimizer.param_groups[0]['lr'] = args.lr * min(1, args.step / (
                        len(dataloader) * args.n_epochs_warmup))

        x = x.requires_grad_(True if args.checkpoint_grads else False).to(args.device)  # requires_grad needed for checkpointing

        optimizer.zero_grad()

        zs, logdet = model.forward(x)
        loss_nll = - model.log_prob_latent(zs, logdet, bits_per_pixel=True).mean(0)
        loss_nll.backward()

        if args.beta_ln > 0:
            zs_noisy = [z.detach() + latent_noise.sample(z.shape) for z in zs]
            x_ln, _ = model.inverse(zs_noisy)
            loss_ln = args.beta_ln * F.mse_loss(x_ln, x).sqrt()
            if loss_ln <= 3:
                loss_ln.backward()
        else:
            loss_ln = 0

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
        optimizer.step()

        # report stats
        if i % args.log_interval == 0:
            report_progress(dataloader, optimizer, writer, epoch, i, loss_nll, loss_ln, tic, args)

        # save and generate
        if args.save_checkpoints and args.step % args.save_interval == 0:
            save_progress(model, optimizer, writer, epoch, args)
        data_loading_timer = time.perf_counter()


def evaluate_progress(model, test_dataloader, epoch, args):
    global best_eval_logprob

    eval_logprob_mean, eval_logprob_std = evaluate(model, test_dataloader, args)
    print(f'Evaluate at epoch {epoch}: bits_x = {eval_logprob_mean:.3f} +/- {eval_logprob_std:.3f}')

    # save best state
    if False and args.on_main_process and eval_logprob_mean > best_eval_logprob:
        best_eval_logprob = eval_logprob_mean
        torch.save({'epoch': epoch,
                    'global_step': args.step,
                    'state_dict': model.state_dict()},
                   os.path.join(args.output_dir, 'best_model_checkpoint.pt'))


def report_progress(dataloader, optimizer, writer, epoch, i, loss_nll, loss_ln, tic, args):
    if args.on_main_process:
        loss = loss_nll + loss_ln
        et = time.time() - tic  # elapsed time
        tt = len(dataloader) * et / (i + 1)  # total time per epoch
        print(f'Epoch: [{epoch + 1}/{args.start_epoch + args.n_epochs}][{i + 1}/{len(dataloader)}]\t'
              f'Step: {args.step}\t'
              f'Time: elapsed {et // 60:.0f}m{et % 60:02.0f}s / total {tt // 60:.0f}m{tt % 60:02.0f}s\t'
              f'Loss {loss.item():.4f}\t'
              )

        writer.add_scalar('loss', loss.item(), args.step)
        writer.add_scalar('loss_nll', loss_nll.item(), args.step)
        if args.beta_ln > 0:
            writer.add_scalar('loss_ln', loss_ln.item(), args.step)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)


def save_progress(model, optimizer, writer, epoch, args):
    # write stats and save checkpoints
    if args.on_main_process:
        if args.step % args.generate_interval == 0:
            # generate samples
            samples = generate(model, n_samples=4, z_stds=[0., 0.25, 0.7, 1.0])
            samples = to_real_images(samples)
            images = make_grid(samples.cpu(), nrow=4, pad_value=1, normalize=True)

            samples_dash = generate(model, n_samples=4, z_stds=4*[1.0])
            samples_dash = to_real_images(samples_dash)
            images_dash = make_grid(samples_dash.cpu(), nrow=4, pad_value=1, normalize=True)

            save_image(images, os.path.join(args.output_dir, 'generated_sample_{}.png'.format(args.step)))
            writer.add_image('generated_sample', images_dash, args.step)

        # save training checkpoint
        torch.save({'epoch': epoch,
                    'global_step': args.step,
                    'state_dict': model.state_dict()},
                   os.path.join(args.output_dir, 'checkpoint.pt'))
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))


def to_real_images(samples):
    samples = transforms.chan_to_complex(samples)
    return fastmri.complex_abs(samples)


@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()
    logprobs = []
    for x, y in dataloader:
        x = x.to(args.device)
        logprobs.append(model.log_prob(x, bits_per_pixel=True))
    logprobs = torch.cat(logprobs, dim=0).to(args.device)
    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.std(0) / math.sqrt(len(dataloader.dataset))
    return logprob_mean, logprob_std


@torch.no_grad()
def generate(model, n_samples, z_stds):
    model.eval()
    print('Generating ...', end='\r')

    samples = []
    for z_std in z_stds:
        sample, _ = model.inverse(batch_size=n_samples, z_std=z_std)
        log_probs = model.log_prob(sample, bits_per_pixel=True)
        samples.append(sample[log_probs.argsort().flip(0)])  # sort by log_prob; flip high (left) to low (right)
    return torch.cat(samples, 0)


@torch.no_grad()
def data_dependent_init(model, args):
    # set up an iterator with batch size = batch_size_init and run through model
    dataloader = get_dataloader_train(args, data_dependent_init=True)
    model(next(iter(dataloader))[0].requires_grad_(True if args.checkpoint_grads else False).to(args.device))
    del dataloader
    return True


def run(command_line_args):
    if command_line_args.restore_file:
        args = utils.dotdict(torch.load(command_line_args.restore_file.replace('checkpoint', 'params'),
                                        map_location='cpu'))
        assert command_line_args.acquisition == args.acquisition, ('Mismatch between provided acquisition and '
                                                                   'acquisition used in checkpoint.')
        args.update(command_line_args.__dict__)
    else:
        config_path = os.path.join(paths.get_config_path(), 'training.yaml')
        args = utils.load_config(config_path)
        args.update(command_line_args.__dict__)

        for param, val in args.items():
            if isinstance(val, dict):
                args[param] = val[args.acquisition]

        n_iter_per_epoch = len(get_mri_dataset_train(args)) // args.batch_size
        args.n_epochs = args.n_iter // n_iter_per_epoch
        args.n_epochs_warmup = args.n_iter_warmup // n_iter_per_epoch

        args.output_dir = os.path.join(paths.get_model_path(), utils.get_timestamp())

    args.input_dims = (2, args.patch_size, args.patch_size)
    args.step = 0  # global step

    writer = None  # init as None in case of multiprocessing; only main process performs write ops

    # setup device
    args.device = utils.get_device(args.local_rank)
    # write ops only when on_main_process
    args.on_main_process = True

    # setup seed
    if args.seed:
        torch.manual_seed(args.seed)
        if args.device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)

    # load data
    train_dataloader = get_dataloader_train(args)

    # load model
    model = Glow(width=args.width,
                 depth=args.depth,
                 n_levels=args.n_levels,
                 input_dims=args.input_dims,
                 affine_scale_low=args.affine_scale_low,
                 unet_coupling=args.unet_coupling,
                 try_to_center=args.try_to_center,
                 checkpoint_grads=args.checkpoint_grads).to(args.device)

    # load optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load checkpoint if provided
    if args.restore_file:
        model_checkpoint = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(
            torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device))
        args.start_epoch = model_checkpoint['epoch']
        args.step = model_checkpoint['global_step']

    # setup writer and outputs
    if args.on_main_process:
        writer = SummaryWriter(log_dir=args.output_dir)

        # save settings
        params_path = os.path.join(args.output_dir, 'params.pt')
        if not os.path.exists(params_path):
            torch.save(dict(args), params_path)

        config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(dict(args))) + \
                 'Num trainable params: {:,.0f}\n\n'.format(sum(p.numel() for p in model.parameters())) + \
                 'Model:\n{}'.format(model)
        config_path = os.path.join(args.output_dir, 'config.txt')
        writer.add_text('model_config', config)
        if not os.path.exists(config_path):
            with open(config_path, 'a') as f:
                print(config, file=f)

    print(f'device: {args.device}')
    print('Performing data dependent initialization. This might take a while...')
    data_dependent_init(model, args)
    low = torch.tensor(-0.5, device=args.device)
    high = torch.tensor(0.5, device=args.device)
    latent_noise = D.Uniform(low, high)

    print(f'Starting training. Tensorboard logs are saved to {args.output_dir}')
    train(model, train_dataloader, optimizer, writer, latent_noise, args)

    if args.on_main_process:
        writer.close()


def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--acquisition',
        choices=('including_fat', 'excluding_fat'),
        required=True,
        type=str,
        help='Which subset of the knee images to train on'
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        help='Which GPU to use'
    )

    parser.add_argument(
        '--restore_file',
        type=str,
        help='Path to the model checkpoint to restore from'
    )

    command_line_args = parser.parse_args()

    run(command_line_args)


if __name__ == '__main__':
    main()
