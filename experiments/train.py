import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from data import NYCTaxiFareDataset
from nn import NYCTaxiFareModel
from visdom import Visdom
import numpy as np


def get_data_loaders(batch_size):
    train_loader = DataLoader(
        NYCTaxiFareDataset('../data/'),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        NYCTaxiFareDataset('../data/', train=False),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def main(config):
    viz = Visdom(port=8098)

    loss_plot = viz.line(X=np.array([0]), Y=np.array([np.nan]))

    run_id = str(int(time.time()))[-7:]
    device = 'cuda'

    model = NYCTaxiFareModel(54, 500, 1)
    model.train()
    model.to(device)

    train_loader, val_loader = get_data_loaders(config.batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    i = 0
    for epoch in range(config.n_epochs):
        for x, y, t in tqdm(train_loader):
            x, y, t = x.to(device), y.to(device), t.to(device)

            z = model(x, y)

            optimizer.zero_grad()
            loss = F.mse_loss(z, t)
            loss.backward()
            optimizer.step()

            viz.line(X=np.array([i]), Y=np.array([loss.data.item()]), update='append',
                     win=loss_plot)

            i += 1
            if i == 20:
                break

        if epoch % 20 == 0:
            model.cpu()
            torch.save(model, f'out/model_{epoch:03d}_{run_id}.pt')

        break


if __name__ == '__main__':
    class config:
        batch_size = 2**16
        lr = 0.01
        n_epochs = 300

    main(config)
