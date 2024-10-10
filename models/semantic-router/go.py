import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder

from mylib import Logger

# sagemaker_config_logger = logging.getLogger('sagemaker.config')
# sagemaker_config_logger.setLevel(logging.WARNING)

#
#
#
class SemanticRouter:
    @staticmethod
    def routes(df):
        for (name, g) in df.groupby('gt', sort=False):
            utterances = g['query'].to_list()
            yield Route(name=name, utterances=utterances)

    def __init__(self, df):
        encoder = OpenAIEncoder()
        routes = list(self.routes(df))
        self.rl = RouteLayer(encoder=encoder, routes=routes)

    def __call__(self, query):
        return self.rl(query).name

def extract(path, split):
    return (pd
            .read_csv(path)
            .query(f'split == "{split}"'))

#
#
#
def func(incoming, outgoing, dpath):
    df = extract(dpath, 'train')
    static = {
        'data': dpath.name,
        'train_n': len(df),
        'train_c': df['gt'].unique().size,
    }
    router = SemanticRouter(df)

    while True:
        sample = incoming.get()
        query = sample['query']
        Logger.info(query)
        outgoing.put({
            **static,
            **sample,
            'pr': router(query),
        })

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args.data,
    )

    with Pool(args.workers, func, initargs):
        df = extract(args.data, 'test')
        for i in df.itertuples(index=False):
            outgoing.put(i._asdict())

        writer = None
        for _ in range(len(df)):
            row = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
