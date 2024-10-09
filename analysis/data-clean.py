import sys
import csv
import logging
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

def func(args):
    (path, pos) = args
    logging.warning(path)

    records = []
    with path.open() as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            for c in ('gt', 'pr'):
                row[c] = int(row[c] == pos)
            records.append(row)

    return records

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=Path)
    arguments.add_argument('--positive', default='query')
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        writer = None

        iterable = ((x, args.positive) for x in args.data.iterdir())
        for i in pool.imap_unordered(func, iterable):
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=i[0])
                writer.writeheader()
            writer.writerows(i)
