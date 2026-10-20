"""TF-IDF retriever over the knowledge base."""
import json
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def stem(word):
    """Light suffix stripping so slow/slowly and drops/dropping match."""
    for suffix in ("ing", "ly", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "ls":
        word = word[:-1]
    return word


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in ENGLISH_STOP_WORDS]


@dataclass
class Hit:
    id: str
    title: str
    text: str
    score: float


class Retriever:
    def __init__(self, docs):
        self.docs = docs
        corpus = [f"{d['title']}. {d['text']}" for d in docs]
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(corpus)

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def search(self, query, k=3):
        scores = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [Hit(self.docs[i]["id"], self.docs[i]["title"], self.docs[i]["text"], float(scores[i]))
                for i in order if scores[i] > 0]
