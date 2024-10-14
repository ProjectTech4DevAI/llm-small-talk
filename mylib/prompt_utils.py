from dataclasses import dataclass

@dataclass
class Prompt:
    role: str
    content: str

    def __str__(self):
        return self.content
