import sys
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
    sizes = set()
    for i in args.data.iterdir():
        df = (pd
              .read_csv(i, usecols=[split])
              .query(f'{split} == "train"'))
        n = len(df)
        if n not in sizes:
            Logger.info(f'{i.name} {n}')
            print(i)
            sizes.add(n)
