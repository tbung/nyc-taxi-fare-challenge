import torch
import torch.nn.functional as F

import numpy as np

from networks import NYCTaxiFareModel


class BaseModel:
    def __init__(self, config):
        self.config = {
            'batch-size': int(2**9),
            'lr': 0.01,
            'n-epochs': 10,
            'save_interval': 10,
            'val_interval': 500,
        }
        self.config.update(config)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_config(self):
        return self.config

    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        mse = 0
        for x, y, t in range(val_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            t = t.to(self.device, non_blocking=True)
            z = self.model(x, y)
            mse += torch.sum((t - z)**2).item()

        return np.sqrt(mse/len(val_loader))

    def save(self, epoch, run_id):
        self.model.cpu()
        torch.save(self.model,
                   f'out/model_{epoch:03d}_{run_id}.pt')
        self.model.to(self.device)


class NetworkModel(BaseModel):
    def __init__(self, config):
        super(NetworkModel, self).__init__(config)
        self.model = NYCTaxiFareModel(54, 500, 1)
        self.model.to(self.device)

        self.loss = F.mse_loss
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.config['lr'])

    def process_batch(self, x, y, t):
        z = self.model(x, y)

        self.optimizer.zero_grad()
        loss = self.loss(z, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.2)
        self.optimizer.step()

        return loss.item()


class SVDKGPModel(BaseModel):
    def __init__(self):
        pass

    def process_batch(self, x, y):
        pass


__models = {
    'network': NetworkModel,
    'svdkgp': SVDKGPModel,
}


def get_model(name):
    return __models[name]


def list_models():
    return __models.keys()
