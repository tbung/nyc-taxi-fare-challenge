import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import smopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.neighbors import KernelDensity

import fire

latex_fonts = {
    'mathtext.fontset': 'cm',      # 'stix'
    'font.family':      'cmss10',  # 'STIXGeneral
    "text.usetex":      True,
    "axes.labelsize":   10,        # LaTeX default is 10pt font.
    "font.size":        10,
    "legend.fontsize":  10,
    "xtick.labelsize":  6,
    "ytick.labelsize":  6,
    # "figure.figsize": figsize(1.0),
    }

colors = {
          'orange': '#ff7f0e',  # used for the INN
          'green':  '#2c993e',
          'purple': '#9243ad',
          'blue':   '#1f77b4',   # used for poitn estimate/MC drop
          'grey':   '#aaaaaa',   # used for gt/prior
         }

matplotlib.rcParams.update(latex_fonts)


data = pd.read_pickle('./data/data_raw.pkl')

# Filter null data
data = data.dropna(how='any', axis='rows')

# Filter negative fare amount
data = data[data['fare_amount'] > 0]
data = data[data['fare_amount'] < 250]

# Filter passenger count
data = data[(data['passenger_count'] <= 6) &
            (data['passenger_count'] >= 1)]

data['year'] = (data['pickup_datetime'].map(lambda x: x.year)
                - 2009)
data['month'] = data['pickup_datetime'].map(lambda x: x.month) - 1
data['weekday'] = data['pickup_datetime'].map(lambda x: x.weekday())
data['quaterhour'] = data['pickup_datetime'].map(
    lambda x: x.hour*4 + x.minute//15
)
data.drop('pickup_datetime', 1, inplace=True)

edata = pd.read_pickle('./data/data_eval.pkl')


def plot_priors():
    # for col in data.iloc[:, :4]:
    col = 'fare_amount'
    print(col)
    col = data[col].iloc[::100]
    x = np.linspace(col.min(), col.max(), 1000)
    kde = KernelDensity(
        kernel='gaussian', bandwidth=0.5
    ).fit(col.values.reshape(-1, 1))
    print('Fitted')
    scores = kde.score_samples(x.reshape(-1, 1))
    print('Scored')

    plt.figure()
    col.hist(bins=100, figsize=(12, 4),
             density=True, alpha=0.3, color='C0')
    plt.plot(x, np.exp(scores), color='C0')
    plt.xlabel('Fare Amount in USD')
    plt.savefig(f'{col.name}_prior.pdf', bbox_inches='tight')


def plot_map():
    # load image of NYC map
    BB = (-75.5, -72.5, 39.5, 42.0)
    nyc_map = plt.imread('./map_-75.5_-72.5_39.5_42.0.png')
    bb_pu = np.array((edata.pickup_longitude.min(), edata.pickup_latitude.min(),
             edata.pickup_longitude.max(), edata.pickup_latitude.max()))
    bb_do = np.array((edata.dropoff_longitude.min(), edata.dropoff_latitude.min(),
             edata.dropoff_longitude.max(), edata.dropoff_latitude.max()))
    # Create a Rectangle patch
    rect_pu = patches.Rectangle(bb_pu[:2], *(bb_pu[2:] - bb_pu[:2]), linewidth=1, edgecolor='r', facecolor='none')
    rect_do = patches.Rectangle(bb_do[:2], *(bb_do[2:] - bb_do[:2]), linewidth=1, edgecolor='r', facecolor='none')

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.scatter(data[::100].pickup_longitude,
               data[::100].pickup_latitude, zorder=1,
               alpha=0.6, c='r', s=1)
    ax.set_xlim((BB[0], BB[1]))
    ax.set_ylim((BB[2], BB[3]))
    ax.imshow(nyc_map, zorder=0, extent=BB)
    ax.add_patch(rect_pu)
    with open('pu_map.png', 'wb') as outfile:
        fig.canvas.print_png(outfile, dpi=1200)

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.scatter(data[::100].dropoff_longitude,
               data[::100].dropoff_latitude, zorder=1,
               alpha=0.6, c='r', s=1)
    ax.set_xlim((BB[0], BB[1]))
    ax.set_ylim((BB[2], BB[3]))
    ax.imshow(nyc_map, zorder=0, extent=BB)
    ax.add_patch(rect_do)
    ax.axis('off')
    with open('do_map.png', 'wb') as outfile:
        fig.canvas.print_png(outfile, dpi=1200)

    fig, axs = plt.subplots(1, 2)
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    # axs[0].set_title('Pickup locations')
    axs[0].imshow(plt.imread('pu_map.png'), extent=BB)
    axs[0].autoscale(False)

    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    # axs[1].set_title('Dropoff locations')
    axs[1].imshow(plt.imread('do_map.png'), extent=BB)
    axs[0].autoscale(False)

    fig.savefig('./locations.pdf', dpi=2400, bbox_inches='tight')


if __name__ == '__main__':
    fire.Fire()
