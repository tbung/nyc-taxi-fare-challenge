import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
from torch.utils.data import DataLoader

import pyro
import pyro.contrib.gp as gp
from sklearn.cluster import MiniBatchKMeans
from collections import OrderedDict
from pathlib import Path
import time
import crayons

from tqdm import tqdm, trange

import numpy as np

from network import TaxiNet
from data import NYCTaxiFareDataset

pyro.set_rng_seed(1234)

runid = str(int(time.time()))[-7:]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = torch.load('../data/data_train/tgt.pt')

search_space = OrderedDict({
    'dim_hidden': [10, 50, 100, 500, 1000, 5000],
    'n_clusters': [1, 10, 50, 100, 500, 1000, 5000],
    'n_layers': list(range(1, 5)),
    'lr': [0.1**i for i in range(1, 8)],
    'batch_size': [2**i for i in range(6, 14)],
    'momentum': np.arange(0, 1, 0.1),
    'p_drop': np.arange(0, 1, 0.1),
})

lower_bound = torch.tensor([min(v) for k, v in search_space.items()],
                           dtype=torch.double)
upper_bound = torch.tensor([max(v) for k, v in search_space.items()],
                           dtype=torch.double)


def to_01(x):
    return (x - lower_bound)/(upper_bound - lower_bound)


def from_01(x):
    return (upper_bound - lower_bound) * x + lower_bound


def pick_initial(N):
    x = []
    for k, v in search_space.items():
        x.append(np.random.choice(v, N))

    return torch.tensor(np.stack(x, axis=1), dtype=torch.double)


def label_params(params):
    res = {}
    for k, v in zip(search_space.keys(), params.squeeze()):
        res[k] = v.item()

    return res


def get_data_loaders(batch_size, data_size):
    train_loader = DataLoader(
        NYCTaxiFareDataset('../data/', size=data_size),
        batch_size=batch_size, shuffle=True,
        num_workers=6, pin_memory=False
    )
    test_set = NYCTaxiFareDataset('../data/', size=data_size, train=False)
    val_loader = DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False
    )
    return train_loader, val_loader


def lower_confidence_bound(x, model, kappa=2):
    mu, variance = model(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    return mu - kappa * sigma


def update_posterior(params, model):
    y_ = train(params).to('cpu', torch.double)
    X = torch.cat([model.X.data, params.data], dim=0)
    y = torch.cat([model.y.data, y_.data.reshape(-1)], dim=0)
    model.set_data(X, y)
    model.optimize()
    return y_


def find_a_candidate(x_init, model):
    # transform x to an unconstrained domain
    constraint = constraints.interval(0., 1)
    x_init = to_01(x_init)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    unconstrained_x = torch.tensor(unconstrained_x_init, requires_grad=True,
                                   dtype=torch.double)
    minimizer = optim.LBFGS([unconstrained_x])

    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        x = from_01(x)
        y = lower_confidence_bound(x, model)
        autograd.backward(x, autograd.grad(y, x))
        return y

    minimizer.step(closure)
    # after finding a candidate in the unconstrained domain,
    # convert it back to original domain.
    x = transform_to(constraint)(unconstrained_x)
    x = from_01(x)
    return x.detach()


def next_x(model, lower_bound=0, upper_bound=1, num_candidates=5):
    candidates = []
    values = []

    x_init = model.X[-1:]

    for i in range(num_candidates):
        x = find_a_candidate(x_init, model)
        y = lower_confidence_bound(x, model)
        candidates.append(x)
        values.append(y)
        x_init = torch.empty_like(x_init).uniform_(lower_bound, upper_bound)
        x_init = from_01(x_init)

    argmin = torch.min(torch.cat(values), dim=0)[1].item()
    return candidates[argmin]


def optimize(n_iter):
    print(crayons.cyan("Obtaining initial samples", bold=True))
    x_init = pick_initial(30)
    y = []
    for x in tqdm(x_init, ncols=80, leave=False):
        y.append(train(x).to(torch.double))

    y = torch.stack(y).cpu()

    gpmodel = gp.models.GPRegression(
        x_init, y,
        gp.kernels.Exponential(input_dim=len(search_space)),
        noise=torch.tensor(0.1), jitter=1.0e-4
    ).to(torch.double)
    print(crayons.cyan("Finding optimal parameters", bold=True))
    for i in range(n_iter):
        xmin = next_x(gpmodel)
        y = update_posterior(xmin, gpmodel)

    return xmin


def train(params, max_epochs=10):
    params = label_params(params.to(torch.float))
    # print(params)

    train_loader, test_loader = get_data_loaders(int(params['batch_size']),
                                                 1000000)

    n_clusters = int(params['n_clusters'])

    if n_clusters > 1:
        # Find clusters
        k = MiniBatchKMeans(n_clusters=n_clusters, compute_labels=False,
                            max_iter=10,
                            max_no_improvement=100, random_state=1234)

        k.fit(d.numpy().reshape(-1, 1))
        clusters = torch.tensor(k.cluster_centers_.squeeze(),
                                dtype=torch.float,
                                device=device)

    model = TaxiNet(dim_out=n_clusters, **params).to(device, torch.float)

    optimizer = optim.SGD(
        model.parameters(), lr=params['lr'],
        momentum=params['momentum']
    )

    for epoch in trange(max_epochs, ncols=80, leave=False):
        for x, y in tqdm(train_loader, ncols=80, leave=False):
            x, y = x.to(device, torch.float), y.to(device, torch.float)

            optimizer.zero_grad()
            y_ = model(x)

            if n_clusters > 1:
                y_ = clusters @ y_.t()

            loss = F.mse_loss(y, y_.squeeze())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()

    with torch.no_grad():
        mse = 0
        for x, y in test_loader:
            x, y = x.to(device, torch.float), y.to(device, torch.float)
            y_ = model(x)

            if n_clusters > 1:
                y_ = clusters @ y_.t()

            mse += torch.sum((y - y_)**2)

        rmse = torch.sqrt(mse/len(test_loader.dataset))

    model.cpu()
    save(model, params, f'out/model_{runid}_rmse{rmse:03.2f}.pt')
    model.to(device)
    print(crayons.red(f"RMSE: {rmse}"))

    return rmse


def save(model, params, file):
    torch.save((model, params), file)


def load(file):
    return torch.load(file)


if __name__ == "__main__":
    outpath = Path('./out')
    outpath.mkdir(exist_ok=True)

    label_params(optimize(50))
