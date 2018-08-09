import pandas as pd

TRAIN_PATH = './train.csv'

# Set columns to most suitable type to optimize for memory usage
types = {'fare_amount': 'float32',
         'pickup_longitude': 'float32',
         'pickup_latitude': 'float32',
         'dropoff_longitude': 'float32',
         'dropoff_latitude': 'float32',
         'passenger_count': 'uint8'}

cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude',
        'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'pickup_datetime']

chunksize = 2_000_000  # 5 million rows at one go. Or try 10 million

train_df = pd.DataFrame()

for i, df_chunk in enumerate(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=types,
                                         chunksize=chunksize)):

    # Each chunk is a corresponding dataframe
    print(f'DataFrame Chunk {i}')

    # Neat trick from
    # https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Slicing off unnecessary components of the datetime and specifying the
    # date format results in a MUCH more efficient conversion to a datetime
    # object.
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(
        df_chunk['pickup_datetime'],
        utc=True, format='%Y-%m-%d %H:%M'
    )

    train_df = train_df.append(df_chunk, ignore_index=True)

# train_df.reset_index(drop=True, inplace=True)
# print(train_df.head())
# train_df.columns = cols
train_df.to_feather('./train.feather')
