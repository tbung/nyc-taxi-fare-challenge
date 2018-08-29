import time
import argparse
import numpy as np

from torch.utils.data import DataLoader
from visdom import Visdom
from tqdm import tqdm

import models
from data import NYCTaxiFareDataset


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


def main(args):
    viz = Visdom(port=8098)

    loss_plot = viz.line(X=np.array([0]), Y=np.array([np.nan]),
                         opts=dict(xlabel='Batch', ylabel='MSE Loss'))
    acc_plot = viz.line(X=np.array([0]), Y=np.array([np.nan]),
                        opts=dict(xlabel='Batch', ylabel='Val Acc'))

    run_id = str(int(time.time()))[-7:]
    device = 'cuda'

    model = models.get_model(args.model)(vars(args))
    config = model.get_config()

    train_loader, val_loader = get_data_loaders(config['batch-size'])

    n_eval = 5*len(train_loader)//100

    for epoch in range(config['n-epochs']):
        for i, (x, y, t) in enumerate(tqdm(train_loader)):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            loss = model.process_batch(x, y, t)

            viz.line(
                X=np.array([i]),
                Y=np.array([loss]),
                update='append',
                win=loss_plot
            )

            if (i+1) % n_eval == 0:
                model.save(epoch, run_id)
                acc = model.eval(val_loader)
                viz.line(X=np.array([i]), Y=np.array([acc]), update='append',
                         win=acc_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=models.list_models(),
                        help='Model to use for training')
    parser.add_argument('--n-epochs', type=int, default=argparse.SUPPRESS,
                        help='')
    parser.add_argument('--save-interval', type=int, default=argparse.SUPPRESS,
                        help='')
    parser.add_argument('--val-interval', type=int, default=argparse.SUPPRESS,
                        help='')
    parser.add_argument('--batch-size', type=int, default=argparse.SUPPRESS,
                        help='')
    parser.add_argument('--lr', type=int, default=argparse.SUPPRESS,
                        help='')

    args = parser.parse_args()

    main(args)
