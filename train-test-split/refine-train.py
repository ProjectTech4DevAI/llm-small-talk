import sys
from argparse import ArgumentParser

import pandas as pd

from mylib import DataSplitter

def refine(df, size, seed):
    splitter = DataSplitter(size, seed)
    for (i, g) in df.groupby('split', sort=False):
        if i == 'train':
            g = (splitter
                 .split(g, 'gt')
                 .query('split == "train"'))
        yield g

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--seed', type=int, default=1234)
    arguments.add_argument('--train-size', type=float, default=0.8)
    args = arguments.parse_args()

    df = (pd
          .read_csv(sys.stdin)
          .assign(seed=args.seed))
    if 0 < args.train_size < 1:
        df = pd.concat(refine(df, args.train_size, args.seed))
    df.to_csv(sys.stdout, index=False)
