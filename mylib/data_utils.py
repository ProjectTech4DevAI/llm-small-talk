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
