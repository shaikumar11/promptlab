"""Ties retrieval, the prompt library and the LLM client together."""
import os
from dataclasses import dataclass, field
from .llm import get_client
from .prompts import default_library
from .retriever import Retriever

NOT_FOUND = "I could not find this in the knowledge base."
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)
    refused: bool = False
    prompt: str = ""


class Assistant:
    def __init__(self, retriever=None, client=None, library=None, min_score=0.12):
        self.retriever = retriever or Retriever.from_file(os.path.join(DATA, "kb.json"))
        self.client = client or get_client()
        self.library = library or default_library()
        self.min_score = min_score

    def ask(self, question, prompt_version="v2", k=3):
        hits = self.retriever.search(question, k)
        if not hits or hits[0].score < self.min_score:
            return Answer(NOT_FOUND, [], True)
        context = "\n".join(f"[{h.id}] {h.title}: {h.text}" for h in hits)
        prompt = self.library.render("qa", prompt_version, context=context, question=question)
        text = self.client.generate(prompt, question=question, hits=hits)
        return Answer(text, [h.id for h in hits], False, prompt)
