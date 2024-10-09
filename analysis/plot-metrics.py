import sys
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--scale-height', type=float, default=2)
    arguments.add_argument('--output', type=Path)
    args = arguments.parse_args()

    groups = (pd
              .read_csv(sys.stdin)
              .groupby('metric', sort=False))
    nrows = groups.ngroups
    (fig, axes) = plt.subplots(nrows=nrows, sharex=True)

    (width, height) = fig.get_size_inches()
    fig.set_size_inches(width, height * args.scale_height)

    for (i, (ax, (m, df))) in enumerate(zip(axes, groups), 1):
         sns.lineplot(
             x='support',
             y='score',
             hue='metric',
             style='metric',
             data=df,
             markers=True,
             legend=False,
             errorbar='pi',
             err_style='bars',
             ax=ax,
         )
         ax.set_xlabel('Training examples' if i == nrows else '')
         ax.set_ylabel(m)
         ax.grid(axis='both', alpha=0.4)

    plt.savefig(args.output, bbox_inches='tight')
