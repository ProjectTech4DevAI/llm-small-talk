import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder

from mylib import Logger, DataReader

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

#
#
#
def func(incoming, outgoing, dpath):
    reader = DataReader(dpath)
    static = {
        'data': str(reader),
        'train_n': len(reader.train),
        'train_c': reader.train['gt'].unique().size,
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
        reader = DataReader(args.data)
        for i in reader.test.itertuples(index=False):
            outgoing.put(i._asdict())

        writer = None
        for _ in range(len(reader.test)):
            row = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
