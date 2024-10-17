import sys
import csv
import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import asdict

from openai import OpenAI

from mylib import (
    Logger,
    Prompt,
    DataReader,
    PromptTimer,
    TestIterator,
)

#
#
#
class SemanticRouter(PromptTimer):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.client = OpenAI()

    def send(self, messages):
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
        )

    def receive(self, response):
        (r, ) = response.choices
        return r.message.content

#
#
#
if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data', type=Path)
    arguments.add_argument('--system-prompt', type=Path)
    args = arguments.parse_args()

    writer = None
    config = json.load(sys.stdin)

    status = config['status']
    if status != 'succeeded':
        raise ValueError(f'{args.data}: {status}')

    router = SemanticRouter(config['fine_tuned_model'])

    data = args.data.joinpath(config['user_provided_suffix'])
    reader = DataReader(data)
    iterable = TestIterator(reader)

    system = Prompt('system', args.system_prompt.read_text().strip())

    for i in iterable:
        user = Prompt('user', i['query'])
        Logger.info(user)

        messages = list(map(asdict, (system, user)))
        response = router(messages)

        row = dict(reader.info, **i)
        row.update(asdict(response))
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row)
            writer.writeheader()
        writer.writerow(row)
