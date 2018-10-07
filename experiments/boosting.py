from data import NYCTaxiFareDataset
import numpy as np
import lightgbm as lgbm
import pandas as pd
import time

runid = str(int(time.time()))[-7:]

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'nthread': 4,
    'num_leaves': 31,
    'learning_rate': 0.029,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'metric': 'rmse',
    'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 10,
    'scale_pos_weight': 1,
    'zero_as_missing': True,
    'seed': 0,
    'num_rounds': 50000
}


def get_datasets(data_size):
    train_set = NYCTaxiFareDataset('../data/', size=data_size)
    val_set = NYCTaxiFareDataset('../data/', size=data_size, train=False)
    train_set = lgbm.Dataset(train_set.features.numpy(),
                             train_set.target.numpy(),
                             silent=False,
                             categorical_feature=list(range(4, 9)))
    val_set = lgbm.Dataset(val_set.features.numpy(), val_set.target.numpy(),
                           silent=False,
                           categorical_feature=list(range(4, 9)))
    return train_set, val_set


def train(params):
    train_set, val_set = get_datasets(8_000_000)
    model = lgbm.train(
        params, train_set=train_set,
        num_boost_round=10000, early_stopping_rounds=500, verbose_eval=50,
        valid_sets=val_set
    )
    model.save_model(f'out/model_{runid}.bst')


def submission():
    model = lgbm.Booster(model_file='./out/model_8655452.bst')

    d = pd.read_csv('../data/test.csv')

    d['pickup_datetime'] = d['pickup_datetime'].str.slice(0, 16)
    d['pickup_datetime'] = pd.to_datetime(d['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    d['year'] = d['pickup_datetime'].map(lambda x: x.year) - 2009
    d['month'] = d['pickup_datetime'].map(lambda x: x.month) - 1
    d['weekday'] = d['pickup_datetime'].map(lambda x: x.weekday())
    d['quaterhour'] = d['pickup_datetime'].map(lambda x: x.hour*4 + x.minute//15)
    d.drop('pickup_datetime', 1, inplace=True)
    mean = np.array([-73.97082, 40.750137, -73.97059, 40.750237])
    std = np.array([0.039113935, 0.030007664, 0.03834897, 0.033217724])
    d.iloc[:, 1:5] = (d.iloc[:, 1:5] - mean) / std
    d['passenger_count'] -= 1

    y = model.predict(d.values[:, 1:])
    out = pd.DataFrame({'key': d['key'].values, 'fare_amount': y})
    out.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    # train(params)
    submission()
