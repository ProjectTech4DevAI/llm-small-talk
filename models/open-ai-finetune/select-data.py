import sys
import collections as cl
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

from mylib import Logger

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=Path)
    arguments.add_argument('--sample', type=int)
    args = arguments.parse_args()

    split = 'split'
    sizes = cl.Counter()

    for i in args.data.iterdir():
        df = (pd
              .read_csv(i, usecols=[split])
              .query(f'{split} == "train"'))

        n = len(df)
        if args.sample is not None and sizes[n] > args.sample:
            continue
        sizes[n] += 1

        Logger.info(f'{i.name} {n}')
        print(i)
