import sys
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--scale-height', type=float, default=2)
    arguments.add_argument('--output', type=Path)
    args = arguments.parse_args()

    df = pd.read_csv(sys.stdin)
    models = df['model'].nunique()

    groups = df.groupby('metric', sort=False)
    nrows = groups.ngroups
    (fig, axes) = plt.subplots(nrows=nrows, sharex=True)

    (width, height) = fig.get_size_inches()
    fig.set_size_inches(width, height * args.scale_height)

    first = 1
    for (i, (ax, (m, g))) in enumerate(zip(axes, groups), first):
        legend = i == first
        sns.lineplot(
            x='train_n',
            y='score',
            hue='model',
            style='model',
            data=g,
            markers=True,
            legend=legend,
            errorbar='pi',
            # err_style='bars',
            ax=ax,
        )
        ax.set_xlabel('Training examples' if i == nrows else '')
        ax.set_ylabel(m)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:0.2f}'))
        ax.grid(axis='both', alpha=0.4)

        if legend:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.25),
                ncols=models,
            )

    plt.savefig(args.output, bbox_inches='tight')
