import sys
import csv
import logging
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split

def split(df, train_size):
    data = train_test_split(
        df,
        train_size=train_size,
        random_state=args.seed,
        stratify=df['classification'],
    )

    for (i, d) in enumerate(data):
        train = int(not i)
        yield d.assign(train=train)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--seed', type=int, default=1234)
    arguments.add_argument('--train-size', type=float, default=0.8)
    arguments.add_argument('--with-ignore', action='store_true')
    arguments.add_argument('--collapse-negatives', action='store_true')
    args = arguments.parse_args()

    ctype = 'classification'
    df = pd.read_csv(sys.stdin, usecols=['question', ctype])
    if not args.with_ignore:
        df = df.query(f'{ctype} != "ignore"')
    if args.collapse_negatives:
        to_replace = ({ctype: x} for x in (r'^(?!.*query).*$', 'small-talk'))
        df = df.replace(*to_replace, regex=True)

    df = pd.concat(split(df, args.train_size))
    df.to_csv(sys.stdout, index=False)
