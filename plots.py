import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fire
import re
from pathlib import Path

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

tag_dict = {'train_loss': 'MSE Loss',
            'test_acc': 'RMSE'}

def plotN(paths, labels, ylim=None):
    records = []
    for path in paths:
        path = Path(path)
        tag = re.match('.*tag-(.*?)\.csv', path.name)[1]
        records.append(pd.read_csv(path, index_col='Step')['Value'])

    fig, ax = plt.subplots()
    for i, record in enumerate(records):
        record.plot(ax=ax, alpha=0.3, color=f'C{i}', label='_nolegend_')
        record.rolling(len(record)//10, min_periods=1).mean().plot(ax=ax,
                                                                   color=f'C{i}',
                                                                   label=labels[i])
    ax.set_ylim(ylim)
    ax.set_ylabel(tag_dict[tag])
    ax.legend()
    plt.savefig(str(Path(paths[0]).stem) + '.pdf', bbox_inches='tight')


def plot(path, ylim=None):
    path = Path(path)
    tag = re.match('.*tag-(.*?)\.csv', path.name)[1]
    print(tag)

    records = pd.read_csv(path, index_col='Step')['Value']

    plt.figure()
    ax = records.plot(alpha=0.3)
    plt.gca().set_prop_cycle(None)
    records.rolling(len(records)//10, min_periods=1).mean().plot(ax=ax)
    ax.set_ylim(ylim)
    ax.set_ylabel(tag_dict[tag])
    plt.savefig(str(path.stem) + '.pdf', bbox_inches='tight')


def plotAll():
    plotN([
        './records/run_network_Sep19_23-35-50-tag-train_loss.csv',
        './records/run_network_noC_Oct06_21-28-03-tag-train_loss.csv',
        './records/run_network_noCE_Oct07_11-38-45-tag-train_loss.csv',
    ],['NN + Embeddings + Clustering', 'NN + Embeddings', 'NN'])
    plotN([
        './records/run_network_Sep19_23-35-50-tag-test_acc.csv',
        './records/run_network_noC_Oct06_21-28-03-tag-test_acc.csv',
        './records/run_network_noCE_Oct07_11-38-45-tag-test_acc.csv',
    ],['NN + Embeddings + Clustering', 'NN + Embeddings', 'NN'])
    plot('./records/run_dgp_Oct04_18-03-13-tag-train_loss.csv')
    plot('./records/run_dgp_Oct04_18-03-13-tag-test_acc.csv')
    plot('./records/run_svdkgp_Oct04_15-33-16-tag-train_loss.csv')
    plot('./records/run_svdkgp_Oct04_15-33-16-tag-test_acc.csv')

if __name__ == '__main__':
    fire.Fire()
