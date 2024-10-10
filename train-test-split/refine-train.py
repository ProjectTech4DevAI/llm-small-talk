import sys
from argparse import ArgumentParser

import pandas as pd

from mylib import DataSplitter

def refine(df, size, seed):
    splitter = DataSplitter(size, seed)
    for (i, g) in df.groupby('train', sort=False):
        if i:
            g = (splitter
                 .split(g, 'gt')
                 .query('train == 1'))
        yield g

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--seed', type=int, default=1234)
    arguments.add_argument('--train-size', type=float, default=0.8)
    args = arguments.parse_args()

    objs = refine(pd.read_csv(sys.stdin), args.train_size, args.seed)
    df = (pd
          .concat(objs)
          .assign(seed=args.seed))
    df.to_csv(sys.stdout, index=False)
