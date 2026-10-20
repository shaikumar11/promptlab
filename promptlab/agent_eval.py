"""Runs the LangGraph agent over the labelled questions with the offline client."""
import json
import os
from .agent import build_graph
from .assistant import Assistant, DATA
from .llm import OfflineClient


def evaluate_agent():
    with open(os.path.join(DATA, "eval_set.json"), encoding="utf-8") as f:
        data = json.load(f)
    graph = build_graph(Assistant(client=OfflineClient()))
    answered = sum(graph.invoke({"question": i["q"]})["action"] == "answered" for i in data["in_scope"])
    escalated = sum(graph.invoke({"question": q})["action"] == "escalated" for q in data["out_of_scope"])
    n, m = len(data["in_scope"]), len(data["out_of_scope"])
    return {"in_scope_answered": f"{answered}/{n}", "out_of_scope_escalated": f"{escalated}/{m}"}


if __name__ == "__main__":
    print(json.dumps(evaluate_agent(), indent=2))
