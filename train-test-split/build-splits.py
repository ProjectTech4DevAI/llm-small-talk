import sys
import csv
from argparse import ArgumentParser

import pandas as pd

from mylib import DataSplitter

def scanf(args, fp):
    usecols = (
        'question',
        'classification',
    )

    reader = csv.DictReader(fp)
    for row in reader:
        (q, c) = map(row.get, usecols)
        if not args.with_ignore and ctype == 'ignore':
            continue
        if args.collapse_negatives and ctype == 'query':
            ctype = 'small-talk'

        yield {
            'query': q,
            'gt': c,
        }

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--seed', type=int, default=1234)
    arguments.add_argument('--train-size', type=float, default=0.8)
    arguments.add_argument('--with-ignore', action='store_true')
    arguments.add_argument('--collapse-negatives', action='store_true')
    args = arguments.parse_args()

    df = pd.DataFrame.from_records(scanf(args, sys.stdin))

    splitter = DataSplitter(args.args.train_size, args.seed)
    splitter.split(df, 'gt').to_csv(sys.stdout, index=False)
