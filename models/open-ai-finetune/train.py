import sys
import json
import itertools as it
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory, NamedTemporaryFile
from dataclasses import dataclass, asdict

import pandas as pd
from openai import OpenAI

from mylib import Logger, DataReader

@dataclass
class Prompt:
    role: str
    content: str

def messages(df, system):
    roles = (
        'system',
        'user',
        'assistant',
    )

    for i in df.itertuples(index=False):
        contents = (system, i.query, i.gt)
        prompts = it.starmap(Prompt, zip(roles, contents))
        yield json.dumps({
            'messages': list(map(asdict, prompts)),
        })

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=Path)
    arguments.add_argument('--system-prompt', type=Path)
    arguments.add_argument('--model', default='gpt-4o-mini-2024-07-18')
    # arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    system = (args
              .system_prompt
              .read_text()
              .strip())
    reader = DataReader(args.data)
    client = OpenAI()

    with TemporaryDirectory() as tmpdir:
        output = Path(tmpdir)
        samples = (output
                   .joinpath(args.data.stem)
                   .with_suffix('.jsonl'))
        samples.write_text('\n'.join(messages(reader.train, system)))

        seed = reader.train['seed'].unique().item()
        training_file = client.files.create(
            file=samples.open('rb'),
            purpose='fine-tune',
        )
        job = client.fine_tuning.jobs.create(
            model=args.model,
            training_file=training_file.id,
            suffix=args.data.name,
            seed=seed,
        )
        Logger.info(job)

        # open ai training
