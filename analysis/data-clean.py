import sys
import csv
import logging
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

def func(incoming, outgoing, args):
    while True:
        (dtype, path) = incoming.get()
        logging.warning(path)

        records = []
        with path.open() as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                row['model'] = dtype
                for c in ('gt', 'pr'):
                    pos = row[c] == args.positive
                    row[c] = int(pos)
                records.append(row)

        outgoing.put(records)

def scan(args):
    for data in args.data.iterdir():
        assert data.is_dir()
        dtype = data.name
        for i in data.iterdir():
            yield (dtype, i)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=Path)
    arguments.add_argument('--positive', default='query')
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args,
    )

    with Pool(args.workers, func, initargs):
        jobs = 0
        for i in scan(args):
            outgoing.put(i)
            jobs += 1

        writer = None
        for _ in range(jobs):
            rows = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
