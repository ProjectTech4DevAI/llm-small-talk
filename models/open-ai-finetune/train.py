import json
import time
import itertools as it
from pathlib import Path
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from dataclasses import dataclass, asdict

from scipy import constants
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
    arguments.add_argument('--wait-time-minutes', type=int, default=10)
    arguments.add_argument('--model', default='gpt-4o-mini-2024-07-18')
    args = arguments.parse_args()

    system = (args
              .system_prompt
              .read_text()
              .strip())
    reader = DataReader(args.data)
    client = OpenAI()

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        samples = (path
                   .joinpath(args.data.stem)
                   .with_suffix('.jsonl'))
        samples.write_text('\n'.join(messages(reader.train, system)))

        with samples.open('rb') as fp:
            training_file = client.files.create(
                file=fp,
                purpose='fine-tune',
            )

    seed = reader.train['seed'].unique().item()
    ft_job = client.fine_tuning.jobs.create(
        model=args.model,
        training_file=training_file.id,
        suffix=args.data.name,
        seed=seed,
    )
    Logger.info(ft_job)

    failure = set([
        'failed',
        'cancelled',
    ])
    default_wait = constants.minute * args.wait_time_minutes

    while True:
        status = client.fine_tuning.jobs.retrieve(ft_job.id)
        if status.status == 'succeeded':
            print(status.to_json(indent=3))
            break
        if status.status in failure:
            raise RuntimeError(f'{status.id}: {status.error}')

        if status.estimated_finish is None:
            wait = default_wait
        else:
            wait = max(status.estimated_finish - time.time(), 10)
        Logger.info(f'{status.id}: waiting {wait}s')
        time.sleep(wait)
