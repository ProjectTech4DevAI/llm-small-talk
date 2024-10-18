import sys
import csv
import logging
import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Queue

import pandas as pd
import sklearn.metrics as mt

GroupKey = cl.namedtuple('GroupKey', 'data, train_n, seed, model')

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
        (f, t) = (counts[x] for x in (self.f, self.t))
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
    _metrics = {
        'Accuracy': Accuracy(),
        'Matthews Corr. Coef.': Matthews(),
        'False Pos. Rate': FalsePositive(),
        'False Neg. Rate': FalseNegative(),
    }

    while True:
        (group, df) = incoming.get()
        logging.warning(group)

        records = []
        (gt, pr) = (df[x] for x in ('gt', 'pr'))
        for (m, f) in _metrics.items():
            score = f(gt, pr)
            rec = group._asdict()
            rec.update({
                'metric': m,
                'score': score,
            })
            records.append(rec)

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
        by = list(GroupKey._fields)
        groups = (pd
                  .read_csv(sys.stdin)
                  .groupby(by, sort=False))
        for (i, g) in groups:
            key = GroupKey(*i)
            outgoing.put((key, g))

        writer = None
        for _ in range(groups.ngroups):
            rows = incoming.get()
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
