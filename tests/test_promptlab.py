import unittest
from promptlab.assistant import Assistant, NOT_FOUND
from promptlab.evaluate import evaluate
from promptlab.llm import OfflineClient
from promptlab.prompts import PromptError, PromptLibrary, PromptTemplate, default_library


class PromptTests(unittest.TestCase):
    def test_render_fills_variables(self):
        out = default_library().render("qa", "v2", context="C", question="Q")
        self.assertIn("Question: Q", out)
        self.assertIn("Context:\nC", out)

    def test_missing_variable_raises(self):
        with self.assertRaises(PromptError):
            default_library().render("qa", "v1", question="Q")

    def test_latest_version_is_default(self):
        self.assertEqual(default_library().get("qa").version, "v2")

    def test_duplicate_version_rejected(self):
        lib = PromptLibrary()
        lib.register(PromptTemplate("x", "v1", "{a}"))
        with self.assertRaises(PromptError):
            lib.register(PromptTemplate("x", "v1", "{a}"))

    def test_unknown_prompt(self):
        with self.assertRaises(PromptError):
            default_library().get("nope")


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.a = Assistant(client=OfflineClient())

    def test_answers_in_scope_with_source(self):
        ans = self.a.ask("How do I reset my forgotten password?")
        self.assertFalse(ans.refused)
        self.assertEqual(ans.sources[0], "password-reset")
        self.assertIn("[password-reset]", ans.text)

    def test_refuses_out_of_scope(self):
        ans = self.a.ask("Give me a recipe for chicken biryani")
        self.assertTrue(ans.refused)
        self.assertEqual(ans.text, NOT_FOUND)

    def test_prompt_version_changes_prompt(self):
        p1 = self.a.ask("VPN keeps disconnecting", "v1").prompt
        p2 = self.a.ask("VPN keeps disconnecting", "v2").prompt
        self.assertNotEqual(p1, p2)
        self.assertIn("Cite the source id", p2)


class EvalTests(unittest.TestCase):
    def test_retrieval_quality_floor(self):
        r = evaluate()
        self.assertGreaterEqual(r["hit_at_3"], 0.9)
        self.assertGreaterEqual(r["out_of_scope_rejected"], 0.75)


if __name__ == "__main__":
    unittest.main()
