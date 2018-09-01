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
        NYCTaxiFareDataset('../data/', size=8000000),
        batch_size=batch_size, shuffle=True,
        num_workers=6, pin_memory=False
    )
    test_set = NYCTaxiFareDataset('../data/', size=8000000, train=False)
    val_loader = DataLoader(
        test_set,
        batch_size=len(test_set)//10, shuffle=False,
        num_workers=1, pin_memory=False
    )
    return train_loader, val_loader


def main(args):
    viz = Visdom(port=8098)

    loss_plot = viz.line(X=np.array([0]), Y=np.array([np.nan]), win='nyc1',
                         opts=dict(xlabel='Batch', ylabel='MSE Loss'))
    acc_plot = viz.line(X=np.array([0]), Y=np.array([np.nan]), win='nyc2',
                        opts=dict(xlabel='Batch', ylabel='Val Acc'))

    run_id = str(int(time.time()))[-7:]
    device = 'cuda'

    train_loader, val_loader = get_data_loaders(2**12)

    model = models.get_model(args.model)(vars(args), train_loader)
    config = model.get_config()

    n_eval = config['test-percent']*len(train_loader)//100

    i = 0
    for epoch in range(config['n-epochs']):
        loss = 0
        for x, y in tqdm(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            loss += model.process_batch(x, y)

            if (i+1) % config['loss-epochs'] == 0:
                viz.line(
                    X=np.array([i]),
                    Y=np.array([loss/config['loss-epochs']]),
                    update='append',
                    win=loss_plot
                )
                loss = 0

            if (i+1) % n_eval == 0:
                model.save(epoch, run_id)
                acc = model.eval(val_loader)
                viz.line(X=np.array([i]), Y=np.array([acc]), update='append',
                         win=acc_plot)
            i += 1


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
    parser.add_argument('--checkpoint', type=str, default=argparse.SUPPRESS,
                        help='')

    args = parser.parse_args()

    main(args)
