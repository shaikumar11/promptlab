"""LLM clients. Groq is used when GROQ_API_KEY is set, otherwise an offline extractive client."""
import json
import os
import re
import urllib.request


class OfflineClient:
    """No API needed. Returns the best matching sentences from the top retrieved document."""
    name = "offline"

    def generate(self, prompt, question, hits):
        q = set(re.findall(r"[a-z]+", question.lower()))
        sentences = re.split(r"(?<=[.])\s+", hits[0].text)
        ranked = sorted(sentences, key=lambda s: -len(q & set(re.findall(r"[a-z]+", s.lower()))))
        chosen = [s for s in sentences if s in ranked[:2]]
        return " ".join(f"{i}. {s} [{hits[0].id}]" for i, s in enumerate(chosen, 1))


class GroqClient:
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key, model="openai/gpt-oss-120b"):
        self.api_key, self.model = api_key, model

    def generate(self, prompt, question=None, hits=None):
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.URL, data=body, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "promptlab/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()


def get_client(provider=None):
    provider = provider or ("groq" if os.getenv("GROQ_API_KEY") else "offline")
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY to use the groq provider")
        return GroqClient(key)
    return OfflineClient()
