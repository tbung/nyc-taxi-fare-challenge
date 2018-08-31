# coding: utf-8
import torch
model = torch.load('out/model_046_5586598.pt')
c = torch.load('../data/clusters.pt')
import pandas as pd
d = pd.read_csv('../data/test.csv')
d['pickup_datetime'] = d['pickup_datetime'].str.slice(0, 16)
d['pickup_datetime'] = pd.to_datetime(d['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
d['year'] = d['pickup_datetime'].map(lambda x: x.year) - 2009
d['month'] = d['pickup_datetime'].map(lambda x: x.month) - 1
d['weekday'] = d['pickup_datetime'].map(lambda x: x.weekday())
d['quaterhour'] = d['pickup_datetime'].map(lambda x: x.hour*4 + x.minute//15)
d.drop('pickup_datetime', 1, inplace=True)
import numpy as np
mean = np.array([-73.97082, 40.750137, -73.97059, 40.750237])
std = np.array([0.039113935, 0.030007664, 0.03834897, 0.033217724])
d.iloc[:, 1:5] = (d.iloc[:, 1:5] - mean) / std
d['passenger_count'] -= 1
y = torch.tensor(d.iloc[:, 5:].values, dtype=torch.long)
x = torch.tensor(d.iloc[:, 1:5].values, dtype=torch.float)
z_ = model(x, y)
z = c @ z_.t()
out = pd.DataFrame({'key': d['key'].values, 'fare_amount': z.data.numpy()})
out.to_csv('./submission.csv', index=False)
