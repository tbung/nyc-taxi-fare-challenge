import torch
import torch.nn.functional as F
import gpytorch

import numpy as np
from tqdm import tqdm

from networks import NYCTaxiFareModel
from gp import DKLModel


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
    def __init__(self, config):
        super(NetworkModel, self).__init__(config)
        self.config.update({
            'batch-size': int(2**8),
            'lr': 0.001,
            'n-epochs': 50,
            'loss-epochs': 5,
            'test-percent': 100
        })
        self.config.update(config)
        self.clusters = torch.load('../data/clusters_32.pt').to(self.device)
        self.model = NYCTaxiFareModel(54, self.clusters.shape[0], softmax=True)
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
            np.exp(np.log(0.1)/28000/3)
        )

    def process_batch(self, x, y, t):
        z = self.model(x, y).squeeze()

        z = self.clusters @ z.t()

        self.optimizer.zero_grad()
        loss = self.loss(z, t)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        mse = 0
        for x, y, t in tqdm(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            t = t.to(self.device, non_blocking=True)
            z = self.model(x, y).squeeze()
            z = self.clusters @ z.t()
            mse += torch.sum((t - z)**2).item()

        return np.sqrt(mse/len(val_loader.dataset))


class SVDKGPModel(BaseModel):
    def __init__(self, config):
        super(SVDKGPModel, self).__init__(config)
        self.clusters = torch.load('../data/clusters_32.pt').to(self.device)
        self.config.update({
            'batch-size': int(2**12),
            'lr': 0.001,
            'n-epochs': 50,
            'loss-epochs': 1,
            'test-percent': 500
        })
        self.config.update(config)
        # self.feature_extractor = torch.load('out/model_002_5729593.pt').to(self.device)
        # del self.feature_extractor.layers[-1]
        self.feature_extractor = NYCTaxiFareModel(54, 32,
                                                  softmax=False).to(self.device)
        self.model = DKLModel(self.feature_extractor, clusters=self.clusters).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            self.device
        )

        self.optimizer = torch.optim.SGD([
            {'params': self.model.parameters()},
            # {'params': self.feature_extractor.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.config['lr'], momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1758, 0.1
        )

        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, 1,
        #     np.exp(np.log(1e-1)/550)
        # )

    def init_mll(self, N):
        self.mll = gpytorch.mlls.VariationalMarginalLogLikelihood(
            self.likelihood, self.model, n_data=N
        )

    def process_batch(self, x, y, t):
        self.optimizer.zero_grad()

        # with gpytorch.settings.use_toeplitz(False):
        z = self.model(x, y)
        loss = -self.mll(z, t)

        loss.backward(retain_graph=False)
        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            z = self.likelihood(self.model(x, y)).mean()
            mse = torch.mean((t - z.mean())**2)

        return mse.item()

    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        self.likelihood.eval()
        mse = 0
        for x, y, t in tqdm(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            t = t.to(self.device, non_blocking=True)
            z = self.likelihood(self.model(x, y)).mean()
            mse += torch.sum((t - z)**2).item()

        self.model.train()
        self.likelihood.train()

        return np.sqrt(mse/len(val_loader.dataset))


__models = {
    'network': NetworkModel,
    'svdkgp': SVDKGPModel,
}


def get_model(name):
    return __models[name]


def list_models():
    return __models.keys()
