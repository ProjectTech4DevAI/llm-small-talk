import sys
import logging
from argparse import ArgumentParser

import pandas as pd
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder

sagemaker_config_logger = logging.getLogger('sagemaker.config')
sagemaker_config_logger.setLevel(logging.WARNING)

def mk_routes(df):
    for (i, g) in df.groupby('classification', sort=False):
        yield Route(name=i, utterances=g['question'].to_list())

def do_routes(layer):
    def f(x):
        return x['question'].apply(lambda y: layer(y).name)

    return f

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--seed', type=int, default=1234)
    arguments.add_argument('--train-size', type=float, default=1)
    arguments.add_argument('--with-positive', action='store_true')
    args = arguments.parse_args()

    ctype = 'classification'

    df = pd.read_csv(sys.stdin)
    if not args.with_positive:
        df = df.query(f'{ctype} != "query"')

    train_test = df.groupby('train', sort=False)
    train = train_test.get_group(True)
    if 0 < args.train_size < 1:
        train = train.sample(frac=args.train_size)

    encoder = OpenAIEncoder()
    routes = list(mk_routes(train))
    rl = RouteLayer(encoder=encoder, routes=routes)

    df = (train_test
          .get_group(False)
          .filter(items=['question', ctype])
          .rename(columns={ctype: 'gt'})
          .assign(pr=do_routes(rl)))
    df.to_csv(sys.stdout, index=False)
