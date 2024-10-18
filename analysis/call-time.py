import sys
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--cutoff', type=float)
    arguments.add_argument('--output', type=Path)
    args = arguments.parse_args()

    df = pd.read_csv(sys.stdin)
    sns.ecdfplot(
        x='duration',
        hue='model',
        data=df,
    )
    plt.xlabel('API response time (sec)')
    plt.grid(axis='both', alpha=0.4)
    if args.cutoff is not None:
        plt.xlim(0, args.cutoff)

    plt.savefig(args.output, bbox_inches='tight')
