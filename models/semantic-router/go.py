import sys
import csv
import logging
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder

sagemaker_config_logger = logging.getLogger('sagemaker.config')
sagemaker_config_logger.setLevel(logging.WARNING)

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
def func(incoming, outgoing, df):
    router = SemanticRouter(df)
    static = {
        'train_n': len(df),
        'train_c': df['gt'].unique().size,
    }

    while True:
        sample = incoming.get()
        query = sample['query']
        logging.warning(query)
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
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    data = (pd
            .read_csv(sys.stdin) # query,gt,split,seed
            .groupby('split', sort=False))

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        data.get_group('train'),
    )

    with Pool(args.workers, func, initargs):
        test = data.get_group('test')
        for i in test.itertuples(index=False):
            outgoing.put(i._asdict())

        writer = None
        for _ in range(len(test)):
            row = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
