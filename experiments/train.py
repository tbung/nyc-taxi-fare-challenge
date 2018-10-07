import time
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np

from torch.utils.data import DataLoader
from visdom import Visdom
from tensorboardX import SummaryWriter
from tqdm import tqdm
import signal

import models
from data import NYCTaxiFareDataset


def worker_init(x):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_data_loaders(batch_size, data_size):
    train_loader = DataLoader(
        NYCTaxiFareDataset('../data/', size=data_size),
        batch_size=batch_size, shuffle=True,
        num_workers=6, pin_memory=False, worker_init_fn=worker_init
    )
    test_set = NYCTaxiFareDataset('../data/', size=data_size, train=False)
    val_loader = DataLoader(
        test_set,
        batch_size=len(test_set)//100, shuffle=False,
        num_workers=4, pin_memory=False, worker_init_fn=worker_init
    )
    return train_loader, val_loader


class Trainer:
    def run(self, args):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(f'runs/{args.model}_{current_time}')

        self.run_id = str(int(time.time()))[-7:]
        device = 'cuda'

        self.model = models.get_model(args.model)(vars(args))
        self.config = self.model.get_config()

        train_loader, val_loader = get_data_loaders(self.config['batch_size'],
                                                    self.config['data_size'])

        n_eval = self.config['test_percent']*len(train_loader)//100

        i = 0
        for epoch in range(self.config['n_epochs']):
            loss = 0
            for x, y in tqdm(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if x.shape[0] != self.config['batch_size']:
                    continue

                loss = np.sqrt(self.model.process_batch(x, y))
                self.writer.add_scalar('train/loss', loss, global_step=i)

                if (i+1) % n_eval == 0:
                    self.save(i)
                    acc = self.model.eval(val_loader)
                    print(f"Test error: {acc:.2f}")
                    self.writer.add_scalar('test/acc', acc, global_step=i)
                i += 1

        self.save()

    def save(self, step=None):
        if not step:
            step = -1
        if self.config['model'] == 'network':
            labels = ['Passengers', 'Year', 'Month', 'Day', 'Quaterour']
            for i, embedding in enumerate(
                    self.model.model.feature_creator.embeddings):
                print(labels[i])
                self.writer.add_embedding(
                    embedding.weight,
                    metadata=list(range(embedding.weight.shape[0])),
                    tag=labels[i],
                    global_step=step
                )

        path = Path(self.writer.file_writer.get_logdir())
        path /= "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path, step)

    def close(self):
        path = Path(self.writer.file_writer.get_logdir())
        path /= "records.json"
        self.writer.export_scalars_to_json(path)
        self.writer.close()


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
    parser.add_argument('--data-size', type=int, default=argparse.SUPPRESS,
                        help='')
    parser.add_argument('--lr', type=float, default=argparse.SUPPRESS,
                        help='')
    parser.add_argument('--checkpoint', type=str, default=argparse.SUPPRESS,
                        help='')
    parser.add_argument('--load', type=str, default=argparse.SUPPRESS,
                        help='')

    args = parser.parse_args()

    trainer = Trainer()

    try:
        trainer.run(args)
    except KeyboardInterrupt:
        trainer.save()
        trainer.close()
        print("User Interrupted")
