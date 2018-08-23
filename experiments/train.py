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
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_set = NYCTaxiFareDataset('../data/', train=False)
    val_loader = DataLoader(
        test_set,
        batch_size=len(test_set)//10, shuffle=False,
        num_workers=1, pin_memory=True
    )
    return train_loader, val_loader


@torch.no_grad()
def eval(model, val_loader):
    model.eval()
    mse = 0
    for x, y, t in tqdm(val_loader, desc='Val'):
        x, y, t = x.cuda(async=True), y.cuda(async=True), t.cuda(async=True)
        z = model(x, y)
        mse += torch.sum((t - z)**2).data.item()

    return np.sqrt(mse/len(val_loader))


def main(config):
    viz = Visdom(port=8098)

    loss_plot = viz.line(X=np.array([0]), Y=np.array([np.nan]),
                         opts=dict(xlabel='Batch', ylabel='MSE Loss'))
    acc_plot = viz.line(X=np.array([0]), Y=np.array([np.nan]),
                        opts=dict(xlabel='Batch', ylabel='Val Acc'))

    run_id = str(int(time.time()))[-7:]
    device = 'cuda'

    model = NYCTaxiFareModel(54, 500, 1)
    model.train()
    model.to(device)

    train_loader, val_loader = get_data_loaders(config.batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                momentum=0.9, weight_decay=1e-4)
    n_eval = 5*len(train_loader)//100

    i = 0
    for epoch in range(config.n_epochs):
        for i, (x, y, t) in enumerate(tqdm(train_loader)):
            x, y, t = x.cuda(async=True), y.cuda(async=True), t.cuda(async=True)

            z = model(x, y)

            optimizer.zero_grad()
            loss = F.mse_loss(z, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()

            viz.line(X=np.array([i]), Y=np.array([loss.data.item()]), update='append',
                     win=loss_plot)

            if (i+1) % n_eval == 0:
                model.cpu()
                torch.save(model, f'out/model_{epoch:03d}_{run_id}.pt')
                model.to(device)
                acc = eval(model, val_loader)
                viz.line(X=np.array([i]), Y=np.array([acc]), update='append',
                         win=acc_plot)



if __name__ == '__main__':
    class config:
        batch_size = 2**9
        lr = 0.01
        n_epochs = 10

    main(config)
