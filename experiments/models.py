import torch
import torch.nn.functional as F
import pyro
import pyro.contrib.gp as gp

import numpy as np
from tqdm import tqdm

from networks import NYCTaxiFareModel


class BaseModel:
    def __init__(self, config):
        self.config = {
            'batch-size': int(2**8),
            'lr': 0.001,
            'n-epochs': 50,
            'loss-epochs': 100,
            'test-percent': 100
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

    def save(self, epoch, run_id):
        self.model.cpu()
        torch.save(self.model,
                   f'out/model_{epoch:03d}_{run_id}.pt')
        self.model.to(self.device)


class NetworkModel(BaseModel):
    def __init__(self, config, *args):
        super(NetworkModel, self).__init__(config)
        self.config.update({
            'batch-size': int(2**8),
            'lr': 0.001,
            'n-epochs': 50,
            'loss-epochs': 5,
            'test-percent': 100
        })
        self.config.update(config)
        # self.clusters = torch.load('../data/clusters_32.pt').to(self.device)
        # self.model = NYCTaxiFareModel(54, self.clusters.shape[0], softmax=True)
        self.model = NYCTaxiFareModel(54, 2, softmax=False)
        self.model.to(self.device)

        self.loss = F.mse_loss
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=1e-4,
            momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1,
            np.exp(np.log(0.1)/1758/3)
        )

    def process_batch(self, x, y):
        z = self.model(x).squeeze().mean(dim=1)

        # z = self.clusters @ z.t()

        self.optimizer.zero_grad()
        loss = self.loss(z, y)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
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
            z = self.model(x).squeeze().mean(dim=1)
            # z = self.clusters @ z.t()
            mse += torch.sum((y - z)**2).item()

        return np.sqrt(mse/len(val_loader.dataset))


class SVDKGPModel(BaseModel):
    def __init__(self, config, train_loader):
        super(SVDKGPModel, self).__init__(config)
        self.clusters = torch.load('../data/clusters_32.pt').to(self.device)
        self.config.update({
            'batch-size': int(2**16),
            'lr': 0.001,
            'n-epochs': 50,
            'loss-epochs': 50,
            'test-percent': 100,
            'n_inducing': 12000,
        })
        self.config.update(config)

        if 'checkpoint' in self.config:
            self.feature_extractor = torch.load(self.config['checkpoint']).to(self.device)
            # del self.feature_extractor.layers[-1]
        else:
            self.feature_extractor = NYCTaxiFareModel(
                54, 2, softmax=False
            ).to(self.device)

        def feature_fn(x):
            return pyro.module("DNN", self.feature_extractor)(x)

        kernel = gp.kernels.RBF(input_dim=2,
                                variance=torch.tensor(0.1)).add(gp.kernels.WhiteNoise(input_dim=2)).warp(iwarping_fn=feature_fn)
        Xu = next(iter(train_loader))[0][-self.config['n_inducing']:]
        likelihood = gp.likelihoods.Gaussian(variance=torch.tensor(0.1))
        self.model = gp.models.VariationalSparseGP(
            X=Xu, y=None, kernel=kernel, Xu=Xu, likelihood=likelihood,
            num_data=len(train_loader.dataset), whiten=True
        ).to(self.device)

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
        self.likelihood.eval()
        mse = 0
        for x, y in tqdm(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            z = self.model(x)
            mse += torch.sum((y - z)**2).item()

        self.model.train()
        self.likelihood.train()

        return np.sqrt(mse/len(val_loader.dataset))

    def save(self, epoch, run_id):
        self.model.cpu()
        pyro.get_param_store().save(f'out/model_{epoch:03d}_{run_id}.pt')
        self.model.to(self.device)


__models = {
    'network': NetworkModel,
    'svdkgp': SVDKGPModel,
}


def get_model(name):
    return __models[name]


def list_models():
    return __models.keys()
