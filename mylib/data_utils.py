import functools as ft
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    _splits = (
        'train',
        'test',
    )

    def __init__(self, train_size, random_state):
        self.train_size = train_size
        self.random_state = random_state

    def split(self, df, stratify):
        return pd.concat(self.train_test(df, stratify))

    def train_test(self, df, stratify):
        data = train_test_split(
            df,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=df[stratify],
        )

        for (s, d) in zip(self._splits, data):
            yield d.assign(split=s)

#
#
#
@dataclass
class DataReader:
    path: Path
    train: pd.DataFrame
    test: pd.DataFrame

    @ft.cached_property
    def info(self):
        return (
            ('data', str(self)),
            ('train_n', len(self.train)),
            ('train_c', self.train['gt'].unique().size),
        )

    def __init__(self, path):
        self.path = path
        groups = (pd
                  .read_csv(self.path)
                  .groupby('split', sort=False))
        (self.train, self.test) = map(groups.get_group, ('train', 'test'))

    def __repr__(self):
        return str(self.path)

    def __str__(self):
        return self.path.name

#
#
#
class DataIterator:
    def __init__(self, df):
        self.df = df

    def __iter__(self):
        for i in self.df.itertuples(index=False):
            yield i._asdict()

class TrainIterator(DataIterator):
    def __init__(self, reader):
        super().__init__(reader.train)

class TestIterator(DataIterator):
    def __init__(self, reader):
        super().__init__(reader.test)
