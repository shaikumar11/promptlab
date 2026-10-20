"""Versioned prompt library. Each prompt has a name, a version and required variables."""
from dataclasses import dataclass
from string import Formatter


class PromptError(Exception):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: str

    @property
    def variables(self):
        return sorted({f for _, f, _, _ in Formatter().parse(self.template) if f})

    def render(self, **values):
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise PromptError(f"{self.name}:{self.version} missing variables {missing}")
        return self.template.format(**values)


class PromptLibrary:
    def __init__(self):
        self._items = {}

    def register(self, prompt):
        key = (prompt.name, prompt.version)
        if key in self._items:
            raise PromptError(f"{prompt.name}:{prompt.version} already registered")
        self._items[key] = prompt

    def get(self, name, version=None):
        versions = sorted(v for n, v in self._items if n == name)
        if not versions:
            raise PromptError(f"unknown prompt {name}")
        version = version or versions[-1]
        if (name, version) not in self._items:
            raise PromptError(f"unknown version {version} for {name}")
        return self._items[(name, version)]

    def render(self, name, version=None, **values):
        return self.get(name, version).render(**values)

    def list(self):
        return sorted(self._items)


def default_library():
    lib = PromptLibrary()
    lib.register(PromptTemplate("qa", "v1",
        "Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"))
    lib.register(PromptTemplate("qa", "v2",
        "You are an IT help desk assistant. Answer ONLY from the context below.\n"
        "Give short numbered steps. Cite the source id in square brackets after each step.\n"
        "If the context does not contain the answer, reply exactly: I could not find this in the knowledge base.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"))
    lib.register(PromptTemplate("summarize", "v1",
        "Summarize the following support ticket in two sentences. Mention the problem and the steps tried.\n\nTicket:\n{ticket}\n\nSummary:"))
    return lib
