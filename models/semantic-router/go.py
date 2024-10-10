import sys
import csv
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
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
@dataclass
class Data:
    train: pd.DataFrame
    test: pd.DataFrame

    def __init__(self, fp, args):
        groups = (pd
                  .read_csv(fp)
                  .groupby('train', sort=False))
        (self.test, self.train) = map(groups.get_group, range(2))
        if 0 < args.train_size < 1:
            self.train = self.train.sample(
                frac=args.train_size,
                random_state=args.seed,
            )

#
#
#
class SemanticRouter:
    @staticmethod
    def routes(df):
        for (n, g) in df.groupby('classification', sort=False):
            u = g['question'].to_list()
            yield Route(name=n, utterances=u)

    def __init__(self, df):
        encoder = OpenAIEncoder()
        routes = list(self.routes(df))
        self.rl = RouteLayer(encoder=encoder, routes=routes)

    def __call__(self, query):
        return self.rl(query).name

#
#
#
def func(incoming, outgoing, df, args):
    _c = 'classification'

    router = SemanticRouter(df)
    static = {
        'rndseed': args.seed,
        'train_n': len(df),
        'train_c': df[_c].unique().size,
    }

    while True:
        query = incoming.get()
        question = query['question']

        logging.warning(question)
        record = dict(static)
        record.update({
            'query': question,
            'gt': query[_c],
            'pr': router(question),
        })
        outgoing.put(record)

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--seed', type=int, default=1234)
    arguments.add_argument('--train-size', type=float, default=1)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    data = Data(sys.stdin, args)

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        data.train,
        args,
    )

    with Pool(args.workers, func, initargs):
        for i in data.test.itertuples(index=False):
            outgoing.put(i._asdict())

        writer = None
        for _ in range(len(data.test)):
            row = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
