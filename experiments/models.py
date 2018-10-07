import torch
import torch.nn.functional as F
import pyro
import pyro.contrib.gp as gp

import numpy as np
from tqdm import tqdm
import gc

from networks import NYCTaxiFareModel, NYCTaxiFareResNet
from gp import DGP
from sklearn.cluster import MiniBatchKMeans
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected


class BaseModel:
    def __init__(self, config):
        self.config = {
            'batch_size': int(2**8),
            'data_size': 8_000_000,
            'lr': 0.001,
            'n_epochs': 50,
            'loss_epochs': 100,
            'test_percent': 100
        }
        self.config.update(config)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_config(self):
        return self.config

    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        mse = 0
        for x, y, t in tqdm(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            t = t.to(self.device, non_blocking=True)
            z = self.model(x, y)
            mse += torch.sum((t - z)**2).item()

        return np.sqrt(mse/len(val_loader.dataset))

    def save(self, path, epoch):
        # self.model.cpu()
        torch.save(self.model,
                   path / f'model_{epoch:07d}.pt')


class NetworkModel(BaseModel):
    def __init__(self, config, *args):
        super(NetworkModel, self).__init__(config)
        self.config.update({
            'batch_size': int(2**8),
            'lr': 0.001,
            'n_epochs': 50,
            'loss_epochs': 100,
            'test_percent': 100
        })
        self.config.update(config)
        d = torch.load('../data/data_train/tgt.pt')
        # Find clusters
        k = MiniBatchKMeans(n_clusters=5000, compute_labels=False,
                            max_no_improvement=100, random_state=1234)

        k.fit(d.numpy().reshape(-1, 1))
        self.clusters = torch.tensor(np.unique(k.cluster_centers_.squeeze()),
                                     dtype=torch.float,
                                     )
        self.clusters = torch.cat(
            [self.clusters, d.min().view(1), d.max().view(1)]
        ).to(self.device)
        self.model = NYCTaxiFareModel(71, self.clusters.shape[0], softmax=True)
        self.model.to(self.device)

        self.loss = F.mse_loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            # weight_decay=1e-4,
            # momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1,
            np.exp(np.log(0.1)/1758/3/2**4)
        )

    def process_batch(self, x, y):
        z = self.model(x).squeeze()

        z = self.clusters @ z.t()

        self.optimizer.zero_grad()
        loss = self.loss(z, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.2)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        mse = 0
        for x, y in tqdm(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            z = self.model(x).squeeze()
            z = self.clusters @ z.t()
            mse += torch.sum((y - z)**2).item()

        return np.sqrt(mse/len(val_loader.dataset))


class ResNetModel(NetworkModel):
    def __init__(self, config, *args):
        super(ResNetModel, self).__init__(config)
        self.config.update({
            'batch_size': int(2**12),
            'lr': 0.000001,
            'n_epochs': 50,
            'loss_epochs': 100,
            'test_percent': 100
        })
        self.config.update(config)

        d = torch.load('../data/data_train/tgt.pt')
        # Find clusters
        k = MiniBatchKMeans(n_clusters=5000, compute_labels=False,
                            max_no_improvement=100, random_state=1234)

        k.fit(d.numpy().reshape(-1, 1))
        self.clusters = torch.tensor(np.unique(k.cluster_centers_.squeeze()),
                                     dtype=torch.float,
                                     )
        self.clusters = torch.cat(
            [self.clusters, d.min().view(1), d.max().view(1)]
        ).to(self.device)

        self.model = NYCTaxiFareResNet(71, self.clusters.shape[0], [3, 4, 6, 3])

        self.model.to(self.device)

        self.loss = F.mse_loss
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=self.config['lr'],
        # )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=1e-4,
            momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1,
            np.exp(np.log(0.1)/28125)
        )

    def process_batch(self, x, y):
        z = self.model(x).squeeze()

        z = self.clusters @ z.t()

        self.optimizer.zero_grad()
        loss = self.loss(z, y)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.2)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()


class INNModel(BaseModel):
    def __init__(self, config, *args):
        super(INNModel, self).__init__(config)
        self.config.update({
            'batch_size': int(2**12),
            'lr': 0.001,
            'n_epochs': 50,
            'loss_epochs': 100,
            'test_percent': 20,
            'data_size': 8_000_000
        })
        self.config.update(config)

        # self.embedder = torch.load(
        #     './runs/network_Sep19_23-35-50/checkpoints/model_999.pt'
        # ).feature_creator

        self.ndim_tot = 10
        self.ndim_x = 1
        self.ndim_y = 9
        self.ndim_z = 1

        inp = InputNode(self.ndim_tot, name='input')

        t1 = Node([inp.out0], rev_multiplicative_layer,
                  {'F_class': F_fully_connected,
                   'F_args': {
                       'batch_norm': True, 'internal_size': 2,
                       # 'dropout': 0.3
                   }})

        # t2 = Node([t1.out0], rev_multiplicative_layer,
        #           {'F_class': F_fully_connected, 'clamp': 2.0,
        #            'F_args': {'dropout': 0.5}})

        # t3 = Node([t2.out0], rev_multiplicative_layer,
        #           {'F_class': F_fully_connected, 'clamp': 2.0,
        #            'F_args': {'dropout': 0.5}})

        outp = OutputNode([t1.out0], name='output')

        nodes = [inp, t1, outp]
        self.model = ReversibleGraphNet(nodes)
        self.model.to(self.device)

        self.loss = F.mse_loss
        self.x_noise_scale = 3e-2
        self.y_noise_scale = 3e-2
        self.zeros_noise_scale = 3e-2

        # relative weighting of losses:
        self.lambd_predict = 3.
        self.lambd_latent = 2.
        self.lambd_rev = 10.

        self.pad_x = torch.zeros(self.config['batch_size'], self.ndim_tot -
                                 self.ndim_x)
        # self.pad_yz = torch.zeros(self.config['batch_size'], self.ndim_tot -
        #                           self.ndim_y - self.ndim_z)

        def MMD_multiscale(x, y):
            xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

            rx = (xx.diag().unsqueeze(0).expand_as(xx))
            ry = (yy.diag().unsqueeze(0).expand_as(yy))

            dxx = rx.t() + rx - 2.*xx
            dyy = ry.t() + ry - 2.*yy
            dxy = rx.t() + ry - 2.*zz

            XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
                          torch.zeros(xx.shape).to(self.device),
                          torch.zeros(xx.shape).to(self.device))

            for a in [0.2, 0.5, 0.9, 1.3]:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1

            return torch.mean(XX + YY - 2.*XY)

        def fit(input, target):
            return torch.mean((input - target)**2)

        self.loss_backward = MMD_multiscale
        self.loss_latent = MMD_multiscale
        self.loss_fit = F.l1_loss

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=1e2,
            # momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1758 * 5,
            0.1
        )

    def process_batch(self, x, y):
        self.loss_factor = 1
        y, x = x.to(self.device), y.to(self.device).reshape(-1, 1)
        # y = self.embedder(y[:, :4], y[:, 4:].long()).detach()

        x_clean = x.clone()
        y_clean = y.clone()
        pad_x = self.zeros_noise_scale * torch.randn(
            self.config['batch_size'], self.ndim_tot - self.ndim_x,
            device=self.device
        )
        # pad_yz = self.zeros_noise_scale * torch.randn(
        #     self.config['batch_size'], self.ndim_tot - self.ndim_y -
        #     self.ndim_z, device=self.device
        # )

        x += self.x_noise_scale * torch.randn(
            self.config['batch_size'], self.ndim_x, dtype=torch.float,
            device=self.device
        )
        y += self.y_noise_scale * torch.randn(
            self.config['batch_size'], self.ndim_y, dtype=torch.float,
            device=self.device
        )

        x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat(
                    (torch.randn(self.config['batch_size'], self.ndim_z,
                                 device=self.device), y), dim=1))

        self.optimizer.zero_grad()

        # Forward step:

        output = self.model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)

        l = 0.5 * self.lambd_predict * self.loss_fit(output[:, self.ndim_z:],
                                                     y[:, self.ndim_z:])

        output_block_grad = torch.cat((output[:, :self.ndim_z],
                                       output[:, -self.ndim_y:].data), dim=1)

        l += self.lambd_latent * self.loss_latent(output_block_grad, y_short)
        l_tot = l.data.item()

        l.backward()

        # Backward step:
        # pad_yz = self.zeros_noise_scale * torch.randn(
        #     self.config['batch_size'], self.ndim_tot - self.ndim_y -
        #     self.ndim_z, device=self.device
        # )
        x = x_clean + self.y_noise_scale * torch.randn(
            self.config['batch_size'], self.ndim_x, device=self.device)
        y = y_clean + self.y_noise_scale * torch.randn(
            self.config['batch_size'], self.ndim_y, device=self.device)

        orig_z_perturbed = (output.data[:, :self.ndim_z] + self.y_noise_scale *
                            torch.randn(
                                self.config['batch_size'], self.ndim_z,
                                device=self.device))
        y_rev = torch.cat((orig_z_perturbed, # pad_yz,
                           y), dim=1)
        y_rev_rand = torch.cat((torch.randn(
            self.config['batch_size'], self.ndim_z, device=self.device
        ), y), dim=1)

        output_rev = self.model(y_rev, rev=True)
        output_rev_rand = self.model(y_rev_rand, rev=True)

        l_rev = (
            self.lambd_rev
            * self.loss_factor
            * self.loss_backward(output_rev_rand[:, :self.ndim_x],
                                 x[:, :self.ndim_x])
        )

        mse = torch.mean((output_rev[:, :self.ndim_x] - x[:, :self.ndim_x])**2)
        # l_rev += self.lambd_predict * self.loss_fit(output_rev, x)
        l_rev += self.lambd_predict * self.loss(output_rev[:, :self.ndim_x],
                                                x[:, :self.ndim_x])
        l_tot += l_rev.data.item()
        l_rev.backward()

        for p in self.model.parameters():
            p.grad.data.clamp_(-50, 50)

        self.optimizer.step()
        self.scheduler.step()


        return mse.item()

    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        mse = 0
        for x, y in tqdm(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            # x = self.embedder(x[:, :4], x[:, 4:].long())
            z = self.model(torch.cat(
                (x,
                 torch.zeros(x.shape[0],
                             self.ndim_tot-x.shape[1],
                             device=self.device)), dim=1
            ), rev=True)[:, 0].squeeze()
            mse += torch.sum((y - z)**2).item()

        return np.sqrt(mse/len(val_loader.dataset))


class SVDKGPModel(BaseModel):
    def __init__(self, config):
        super(SVDKGPModel, self).__init__(config)
        gps = torch.load('../data/data_train/gps.pt')
        cat = torch.load('../data/data_train/cat.pt')
        d = torch.cat((gps, cat.float()), dim=1)
        # Find clusters
        k = MiniBatchKMeans(n_clusters=500, compute_labels=False,
                            verbose=True,
                            max_no_improvement=10, random_state=1234)

        k.fit(d.numpy())
        self.clusters = torch.tensor(np.unique(k.cluster_centers_, axis=0),
                                     dtype=torch.float, device=self.device
                                     )
        del gps, cat, d, k
        gc.collect()
        self.config.update({
            'batch_size': int(2**14),
            'lr': 0.01,
            'n_epochs': 50,
            'loss_epochs': 50,
            'test_percent': 100,
            'n_inducing': 8000,
        })
        self.config.update(config)

        if 'checkpoint' in self.config:
            self.feature_extractor = torch.load(self.config['checkpoint']).to(self.device)
            # del self.feature_extractor.layers[-1]
        else:
            self.feature_extractor = NYCTaxiFareModel(
                71, 2, softmax=False
            ).to(self.device)

        if 'load' in self.config:
            pyro.get_param_store().load(self.config['load'],
                                        map_location=self.device)

        def feature_fn(x):
            return pyro.module("DNN", self.feature_extractor,
                               update_module_params=True)(x)

        kernel = gp.kernels.RBF(input_dim=2,
                                variance=torch.tensor(0.1)).add(gp.kernels.WhiteNoise(input_dim=1)).warp(iwarping_fn=feature_fn)
        # Xu = self.feature_extractor(self.clusters).reshape(-1, 1)
        Xu = self.clusters
        likelihood = gp.likelihoods.Gaussian(variance=torch.tensor(0.1))
        self.model = gp.models.VariationalSparseGP(
            X=Xu, y=None, kernel=kernel, Xu=Xu, likelihood=likelihood,
            num_data=self.config['data_size'], whiten=True
        ).to(self.device)

        def model():
            c = pyro.sample("clusters", pyro.distributions.Uniform(0, 250))
            y = self.model.model()
            return self.c @ y

        def guide():
            c = pyro.param("clusters", torch.ones(500),
                           constraint=pyro.constraints.positive)
            y = self.model.guide()

        optimizer = pyro.optim.Adam({'lr': self.config['lr']})
        elbo = pyro.infer.Trace_ELBO()
        self.svi = pyro.infer.SVI(self.model.model, self.model.guide,
                                  optimizer, elbo)

    def process_batch(self, x, y):
        self.model.set_data(x, y)
        loss = self.svi.step()

        with torch.no_grad():
            z, _ = self.model(x)
            mse = torch.mean((y - z)**2)

        return mse.item()

    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        mse = 0
        for x, y in tqdm(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            z, _ = self.model(x)
            mse += torch.sum((y - z)**2).item()

        self.model.train()

        return np.sqrt(mse/len(val_loader.dataset))

    def save(self, path, epoch):
        self.model.cpu()
        pyro.get_param_store().save(path / f'model_{epoch:07d}.pyro')
        self.model.to(self.device)


class DGPModel(BaseModel):
    def __init__(self, config):
        super(DGPModel, self).__init__(config)
        gps = torch.load('../data/data_train/gps.pt')
        cat = torch.load('../data/data_train/cat.pt')
        d = torch.cat((gps, cat.float()), dim=1)
        # Find clusters
        k = MiniBatchKMeans(n_clusters=500, compute_labels=False,
                            verbose=True,
                            max_no_improvement=10, random_state=1234)

        k.fit(d.numpy())
        self.clusters = torch.tensor(np.unique(k.cluster_centers_, axis=0),
                                     dtype=torch.float, device=self.device
                                     )
        del gps, cat, d, k
        gc.collect()
        self.config.update({
            'batch_size': int(2**14),
            'lr': 0.01,
            'n_epochs': 50,
            'loss_epochs': 10,
            'test_percent': 100,
            'n_inducing': 16000,
        })
        self.config.update(config)

        if 'load' in self.config:
            pyro.get_param_store().load(self.config['load'],
                                        map_location=self.device)

        Xu = self.clusters

        self.model = DGP(5, Xu, self.config['data_size'])

        # for layer in self.model.layers:
        #     layer.u_scale_tril.data *= 1e-2

        optimizer = pyro.optim.Adam({'lr': self.config['lr']})
        elbo = pyro.infer.Trace_ELBO()
        self.svi = pyro.infer.SVI(self.model.model,
                                  self.model.guide,
                                  optimizer, elbo)

    def process_batch(self, x, y):
        loss = self.svi.step(x, y)

        with torch.no_grad():
            z, _ = self.model(x)
            mse = torch.mean((y - z)**2)

        return mse.item()

    @torch.no_grad()
    def eval(self, val_loader):
        mse = 0
        for x, y in tqdm(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            z, _ = self.model(x)
            mse += torch.sum((y - z)**2).item()

        return np.sqrt(mse/len(val_loader.dataset))

    def save(self, path, epoch):
        self.model.cpu()
        pyro.get_param_store().save(path / f'model_{epoch:07d}.pyro')
        self.model.to(self.device)


__models = {
    'network': NetworkModel,
    'resnet': ResNetModel,
    'svdkgp': SVDKGPModel,
    'dgp': DGPModel,
    'inn': INNModel
}


def get_model(name):
    return __models[name]


def list_models():
    return __models.keys()
