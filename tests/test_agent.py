import unittest
from promptlab.agent import build_graph, classify_priority
from promptlab.assistant import Assistant
from promptlab.llm import OfflineClient


class StubClient:
    """Returns the given replies one by one and records the prompts it receives."""
    name = "stub"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, question=None, hits=None):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else self.replies_last

    replies_last = "no citation here"


def run(question, client):
    return build_graph(Assistant(client=client)).invoke({"question": question})


class AgentTests(unittest.TestCase):
    def test_answers_grounded_question(self):
        out = run("How do I reset my forgotten password?", OfflineClient())
        self.assertEqual(out["action"], "answered")
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(out["trace"][:3], ["classify:low", "retrieve:found", "draft:qa-v1"])

    def test_escalates_out_of_scope(self):
        out = run("Give me a recipe for chicken biryani", OfflineClient())
        self.assertEqual(out["action"], "escalated")
        self.assertEqual(out["ticket"]["level"], "L2")
        self.assertNotIn("draft:qa-v1", out["trace"])

    def test_retry_uses_stricter_prompt_when_ungrounded(self):
        client = StubClient(["Try restarting.", "1. Reset it from the sign-in page. [password-reset]"])
        out = run("How do I reset my forgotten password?", client)
        self.assertEqual(out["action"], "answered")
        self.assertEqual(out["attempts"], 2)
        self.assertEqual(out["prompt_version"], "v2")
        self.assertIn("Cite the source id", client.prompts[1])
        self.assertNotIn("Cite the source id", client.prompts[0])

    def test_escalates_after_two_ungrounded_attempts(self):
        client = StubClient(["no", "still no", "ticket summary"])
        out = run("How do I reset my forgotten password?", client)
        self.assertEqual(out["action"], "escalated")
        self.assertEqual(out["attempts"], 2)

    def test_priority_rules(self):
        self.assertEqual(classify_priority("I lost my work laptop"), "high")
        self.assertEqual(classify_priority("How do I install new software?"), "low")
        self.assertEqual(classify_priority("My printer shows offline"), "medium")


if __name__ == "__main__":
    unittest.main()
