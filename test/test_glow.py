"""Taken from https://github.com/kamenbliznashki/normalizing_flows"""
import unittest
from unittest.mock import MagicMock
from warnings import filterwarnings

import torch
import torch.distributions as D
from torch.utils.data import DataLoader, Dataset

from kernprior.train_glow import train_epoch
from kernprior.glow import Actnorm, Invertible1x1Conv, AffineCoupling, Squeeze, Split, FlowStep, FlowLevel, Glow

args = MagicMock()
args.batch_size = 100
args.device = torch.device('cpu')

NORM_TOL = 1e-4  # tolerance for difference in vector norms

torch.manual_seed(200)

# --------------------
# Test invertibility and log dets of individual layers
# --------------------


def test_layer(layer, input_dims, norm_tol=NORM_TOL):
    x = torch.randn(input_dims)
    z, logdet = layer(x)
    x_recon, inv_logdet = layer.inverse(z)
    d_data = (x_recon - x).norm()
    d_logdet = (logdet + inv_logdet).norm()
    assert d_data < norm_tol, f'Data reconstruction fail - norm of difference = {d_data}.'
    assert d_logdet < norm_tol, f'Log determinant inversion fail. - norm of difference = {d_logdet}'


class TestFlowLayers(unittest.TestCase):
    def test_actnorm(self):
        test_layer(Actnorm(param_dim=(1, 3, 1, 1)), input_dims=(args.batch_size, 3, 50, 50))

    def test_invertible1x1conv(self):
        test_layer(Invertible1x1Conv(n_channels=24), input_dims=(args.batch_size, 24, 50, 50), norm_tol=1e-3)

    def test_affinecoupling(self):
        test_layer(AffineCoupling(n_channels=4, width=12, affine_scale_low=0.5, unet_coupling=False), input_dims=(args.batch_size, 4, 50, 50), norm_tol=5e-4)

    def test_squeeze(self):
        net = Squeeze()
        x = torch.rand(args.batch_size, 12, 20, 30)
        x_recon = net.inverse(net(x))
        y = net(net.inverse(x))
        assert torch.allclose(x, x_recon), 'Data reconstruction failed.'
        assert torch.allclose(x, y)

    def test_split(self):
        net = Split(n_channels=10)
        x = torch.randn(args.batch_size, 10, 20, 30)
        x1, z2, logdet = net(x)
        x_recon, inv_logdet = net.inverse(x1, z2)
        d_data = (x_recon - x).norm()
        d_logdet = (logdet + inv_logdet).norm()
        assert d_data < 1e-4, 'Data reconstruction fail - norm of difference = {}.'.format(d_data)
        assert d_logdet < 1e-4, 'Log determinant inversion fail. - norm of difference = {}'.format(d_logdet)

    def test_flowstep(self):
        test_layer(FlowStep(n_channels=4, width=12, affine_scale_low=0.5, unet_coupling=False), input_dims=(args.batch_size, 4, 50, 50), norm_tol=1e-3)

    def test_flowlevel(self):
        net = FlowLevel(n_channels=3, width=12, depth=2, affine_scale_low=0.5, unet_coupling=False)
        x = torch.randn(args.batch_size, 3, 32, 32)
        x1, z2, logdet = net(x)
        x_recon, inv_logdet = net.inverse(x1, z2)
        d_data = (x_recon - x).norm()
        d_logdet = (logdet + inv_logdet).norm()
        assert d_data < 5e-4, 'Data reconstruction fail - norm of difference = {}.'.format(d_data)
        assert d_logdet < 5e-4, 'Log determinant inversion fail. - norm of difference = {}'.format(d_logdet)

    def test_glow(self):
        net = Glow(width=12, depth=3, n_levels=3, affine_scale_low=0.5)
        x = torch.randn(args.batch_size, 3, 32, 32)
        zs, _ = net(x)
        x_recon, _ = net.inverse(zs)
        d_data = (x_recon - x).norm()
        assert d_data < 1e-3, 'Data reconstruction fail - norm of difference = {}.'.format(d_data)


# --------------------
# Test Glow
# --------------------

# Generate a dataset from a 2-dim Gaussian distribution and expand to `image size` of (3, 32, 32)

class ToyDistribution(D.Distribution):
    def __init__(self, flip_var_order):
        super().__init__()
        self.flip_var_order = flip_var_order
        self.p_x2 = D.Normal(0, 4)
        self.p_x1 = lambda x2: D.Normal(0.25 * x2**2, 1)

    def rsample(self, sample_shape=torch.Size()):
        x2 = self.p_x2.sample(sample_shape)
        x1 = self.p_x1(x2).sample()
        if self.flip_var_order:
            return torch.stack((x2, x1), dim=-1).expand(3,-1,-1)
        else:
            return torch.stack((x1, x2), dim=0).repeat(16,1).expand(3,-1,-1)

    def log_prob(self, value):
        if self.flip_var_order:
            value = value.flip(1)
        return self.p_x1(value[:,1]).log_prob(value[:,0]) + self.p_x2.log_prob(value[:,1])


class TOY(Dataset):
    def __init__(self, dataset_size=2500, flip_var_order=False):
        self.input_size = 32
        self.label_size = 1
        self.dataset_size = dataset_size
        self.base_dist = ToyDistribution(flip_var_order)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.base_dist.sample((32,)), torch.zeros(self.label_size)


class TestGlowUntrained(unittest.TestCase):
    def setUp(self):
        def test_kl(model):
            n_samples = 1000
            data = model.base_dist.sample((n_samples,3,32,32)).squeeze()
            zs, logdet = model(data)
            recon_data, _ = model.inverse(zs)
            recon_dist = D.Normal(recon_data.mean(0), recon_data.var(0).sqrt())
            kl = D.kl.kl_divergence(recon_dist, model.base_dist).mean()
        self.test_kl = test_kl

    def test_glow_depth_1_levels_1(self):
        # 1. sample data; 2. run model forward and reverse; 3. reconstruct data; 4. measure KL between Gaussian fitted to the data and the base distribution
        self.test_kl(Glow(width=12, depth=1, n_levels=1))

    def test_glow_depth_2_levels_2(self):
        # 1. sample data; 2. run model forward and reverse; 3. reconstruct data; 4. measure KL between Gaussian fitted to the data and the base distribution
        self.test_kl(Glow(width=12, depth=2, n_levels=2))


class TestGlowTrained(unittest.TestCase):
    def setUp(self):
        filterwarnings('ignore',
                       category=UserWarning,
                       message="<class 'test_glow.ToyDistribution'> does not define `arg_constraints`.")

        args_ = MagicMock()
        args_.device = torch.device('cpu')
        args_.n_epochs_warmup = 10
        args_.step = 5
        args_.world_size = 1
        args_.lr = 1e-4
        args_.beta_ln = 0
        data_loader_ = DataLoader(TOY(), batch_size=100)

        @torch.no_grad()
        def _test_trained_model(model, data_loader):
            data, _ = next(iter(data_loader))
            zs, logdet = model(data)
            zs = torch.cat([z.flatten(1) for z in zs], dim=1)  # flatten the z's and concat
            zs_dist = D.Normal(zs.mean(0), zs.std(0))
            kl = D.kl.kl_divergence(zs_dist, model.base_dist).mean()

        def _train(model, n_steps):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
            latent_noise = D.Uniform(-0.5, 0.5)
            _test_trained_model(model, data_loader_)
            for _ in range(n_steps):
                train_epoch(model, data_loader_, optimizer, MagicMock(), 0, latent_noise, args_)
            _test_trained_model(model, data_loader_)

        self._train = _train

    def test_glow_1_1(self):
        model = Glow(width=12, depth=1, n_levels=1)
        self._train(model, 3)

    def test_glow_3_3(self):
        model = Glow(width=24, depth=3, n_levels=3)
        self._train(model, 3)


if __name__ == '__main__':
    unittest.main()
