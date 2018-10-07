import pandas as pd
import torch
import numpy as np
import fire

from sklearn.cluster import MiniBatchKMeans
import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def submission(mod, file):
    d = torch.load('../data/data_train/tgt.pt')
    # Find clusters
    k = MiniBatchKMeans(n_clusters=5000, compute_labels=False,
                        max_no_improvement=100, random_state=1234)

    k.fit(d.numpy().reshape(-1, 1))
    clusters = torch.tensor(np.unique(k.cluster_centers_.squeeze()),
                            dtype=torch.float,
                            )
    clusters = torch.cat(
        [clusters, d.min().view(1), d.max().view(1)]
    ).to(device)

    model = models.get_model(mod)({'load': file})
    model.model.to(device)

    d = pd.read_csv('../data/test.csv')

    d['pickup_datetime'] = d['pickup_datetime'].str.slice(0, 16)
    d['pickup_datetime'] = pd.to_datetime(d['pickup_datetime'], utc=True,
                                          format='%Y-%m-%d %H:%M')
    d['year'] = d['pickup_datetime'].map(lambda x: x.year) - 2009
    d['month'] = d['pickup_datetime'].map(lambda x: x.month) - 1
    d['weekday'] = d['pickup_datetime'].map(lambda x: x.weekday())
    d['quaterhour'] = d['pickup_datetime'].map(lambda x: x.hour*4 +
                                               x.minute//15)
    d.drop('pickup_datetime', 1, inplace=True)

    mean = np.array([-73.97082, 40.750137, -73.97059, 40.750237])
    std = np.array([0.039113935, 0.030007664, 0.03834897, 0.033217724])
    d.iloc[:, 1:5] = (d.iloc[:, 1:5] - mean) / std
    d['passenger_count'] -= 1

    x = torch.tensor(d.iloc[:, 1:].values, dtype=torch.float, device=device)
    z, _ = model.model(x)
    # z = clusters @ z_.t()
    out = pd.DataFrame({'key': d['key'].values, 'fare_amount':
                        z.data.cpu().numpy()})
    out.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    fire.Fire(submission)
