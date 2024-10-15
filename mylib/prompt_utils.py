import timeit
from dataclasses import dataclass

@dataclass
class Prompt:
    role: str
    content: str

    def __str__(self):
        return self.content

@dataclass
class ModelResponse:
    pr: str
    duration: float

class PromptTimer:
    def __call__(self, messages):
        start = timeit.default_timer()
        response = self.send(messages)
        stop = timeit.default_timer()
        pr = self.receive(response)

        return ModelResponse(pr, stop - start)

    def send(self, messages):
        raise NotImplementedError()

    def receive(self, response):
        raise NotImplementedError()
