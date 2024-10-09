import sys
import csv
import logging
import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd
import sklearn.metrics as mt

#
#
#
class Metric:
    def __call__(self, gt, pr):
        raise NotImplementedError()

class Accuracy(Metric):
    def __call__(self, gt, pr):
        return mt.accuracy_score(gt, pr)

class Matthews(Metric):
    def __call__(self, gt, pr):
        return mt.matthews_corrcoef(gt, pr)

class RateMetric(Metric):
    def __init__(self, f, t):
        super().__init__()
        (self.f, self.t) = (f, t)

    def __call__(self, gt, pr):
        counts = cl.Counter(self.pos_neg(gt, pr))
        (f, t) = map(counts.get, (self.f, self.t))
        return f / (f + t)

    def pos_neg(self, gt, pr):
        for (g, p) in zip(gt, pr):
            prefix = 't' if g == p else 'f'
            suffix = 'p' if g else 'n'
            yield prefix + suffix

class FalsePositive(RateMetric):
    def	__init__(self):
        super().__init__('fp', 'tn')

class FalseNegative(RateMetric):
    def	__init__(self):
        super().__init__('fn', 'tp')

#
#
#
def func(incoming, outgoing):
    metrics = {
        'Accuracy': Accuracy(),
        'Matthews Corr. Coef.': Matthews(),
        'False Pos. Rate': FalsePositive(),
        'False Neg. Rate': FalseNegative(),
    }

    while True:
        (group, df) = incoming.get()
        logging.warning(group)

        records = []
        for (i, (_, g)) in enumerate(df.groupby('rndseed', sort=False)):
            assert group == g['train_n'].unique().item()
            (gt, pr) = (g[x] for x in ('gt', 'pr'))
            for (k, v) in metrics.items():
                score = v(gt, pr)
                records.append({
                    'seed': i,
                    'support': group,
                    'metric': k,
                    'score': score,
                })

        outgoing.put(records)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
    )

    with Pool(args.workers, func, initargs):
        groups = (pd
                  .read_csv(sys.stdin)
                  .groupby('train_n', sort=False))
        for g in groups:
            outgoing.put(g)

        writer = None
        for _ in range(groups.ngroups):
            rows = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
