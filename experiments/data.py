import torch.utils.data as data
import pandas as pd
import numpy as np
from pathlib import Path
import torch


class NYCTaxiFareDataset(data.Dataset):
    raw_file = 'data_raw.feather'
    raw_eval = 'eval.feather'
    train_folder = 'data_train'
    test_folder = 'data_test'
    eval_folder = 'data_eval'
    gps_file = 'gps.pt'
    cat_file = 'cat.pt'
    target_file = 'tgt.pt'

    mean = np.array([-73.97082, 40.750137, -73.97059, 40.750237])
    std = np.array([0.039113935, 0.030007664, 0.03834897, 0.033217724])
    min_year = 2009

    def __init__(self, root, train=True):
        self.root = Path(root)
        self.train = train
        self.testsplit = 0.05

        if (not (self.root / self.train_folder).exists()
                or not (self.root / self.test_folder).exists()
                or not (self.root / self.eval_folder).exists()):
            self.preprocess()

        if self.train:
            self.target = torch.load(self.root / self.train_folder /
                                     self.target_file)
            self.gps = torch.load(self.root / self.train_folder /
                                  self.gps_file)
            self.categorical = torch.load(self.root / self.train_folder /
                                          self.cat_file)
        else:
            self.target = torch.load(self.root / self.test_folder /
                                     self.target_file)
            self.gps = torch.load(self.root / self.test_folder /
                                  self.gps_file)
            self.categorical = torch.load(self.root / self.test_folder /
                                          self.cat_file)

    def __getitem__(self, index):
        return (self.gps[index],
                self.categorical[index],
                self.target[index])

    def __len__(self):
        return self.target.shape[0]

    def preprocess(self):
        print("Preprocessing data")
        (self.root / self.train_folder).mkdir()
        (self.root / self.test_folder).mkdir()
        (self.root / self.eval_folder).mkdir()

        data = pd.read_feather(self.root / self.raw_file)
        edata = pd.read_feather(self.root / self.raw_eval)

        # Filter null data
        data = data.dropna(how='any', axis='rows')

        # Filter negative fare amount
        data = data[data['fare_amount'] >= 0]

        # Filter passenger count
        data = data[(data['passenger_count'] <= 6) &
                    (data['passenger_count'] >= 1)]
        data['passenger_count'] -= 1
        edata['passenger_count'] -= 1

        # Convert datetime to usable data
        data['year'] = (data['pickup_datetime'].map(lambda x: x.year)
                        - self.min_year)
        data['month'] = data['pickup_datetime'].map(lambda x: x.month) - 1
        data['weekday'] = data['pickup_datetime'].map(lambda x: x.weekday())
        data['quaterhour'] = data['pickup_datetime'].map(
            lambda x: x.hour*4 + x.minute//15
        )
        data.drop('pickup_datetime', 1, inplace=True)

        edata['year'] = (edata['pickup_datetime'].map(lambda x: x.year)
                         - self.min_year)
        edata['month'] = edata['pickup_datetime'].map(lambda x: x.month) - 1
        edata['weekday'] = edata['pickup_datetime'].map(lambda x: x.weekday())
        edata['quaterhour'] = edata['pickup_datetime'].map(
            lambda x: x.hour*4 + x.minute//15
        )
        edata.drop('pickup_datetime', 1, inplace=True)

        # Filter location data
        lx = data['pickup_longitude']
        ly = data['dropoff_longitude']
        px = data['pickup_latitude']
        py = data['dropoff_latitude']

        elx = edata['pickup_longitude']
        ely = edata['dropoff_longitude']
        epx = edata['pickup_latitude']
        epy = edata['dropoff_latitude']

        data = data[
            (lx <= np.ceil(elx.max())) &
            (lx >= np.floor(elx.min())) &
            (ly <= np.ceil(ely.max())) &
            (ly >= np.floor(ely.min())) &
            (px <= np.ceil(epx.max())) &
            (px >= np.floor(epx.min())) &
            (py <= np.ceil(epy.max())) &
            (py >= np.floor(epy.min()))
        ]

        # Normalize data
        data.iloc[:, 1:5] = (data.iloc[:, 1:5] - self.mean) / self.std
        edata.iloc[:, 0:4] = (edata.iloc[:, 0:4] - self.mean) / self.std

        # train-test split
        t = int(self.testsplit * len(data))

        print("Writing train files")
        torch.save(
            torch.tensor(data.iloc[t:, 0].values, dtype=torch.float),
            self.root / self.train_folder / self.target_file
        )
        torch.save(
            torch.tensor(data.iloc[t:, 1:5].values, dtype=torch.float),
            self.root / self.train_folder / self.gps_file
        )
        torch.save(
            torch.tensor(data.iloc[t:, 5:].values, dtype=torch.long),
            self.root / self.train_folder / self.cat_file
        )

        print("Writing test files")
        torch.save(
            torch.tensor(data.iloc[:t, 0].values, dtype=torch.float),
            self.root / self.test_folder / self.target_file
        )
        torch.save(
            torch.tensor(data.iloc[:t, 1:5].values, dtype=torch.float),
            self.root / self.test_folder / self.gps_file
        )
        torch.save(
            torch.tensor(data.iloc[:t, 5:].values, dtype=torch.long),
            self.root / self.test_folder / self.cat_file
        )

        print("Writing eval file")
        torch.save(
            torch.tensor(edata.iloc[:, :4].values, dtype=torch.float),
            self.root / self.eval_folder / self.gps_file
        )
        torch.save(
            torch.tensor(edata.iloc[:, 4:].values, dtype=torch.long),
            self.root / self.eval_folder / self.cat_file
        )
