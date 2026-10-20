"""TicketFlow: a LangGraph support agent built on the PromptLab retriever and prompt library.

Flow: classify -> retrieve -> (draft -> check -> respond) or escalate
The check node verifies the answer cites a retrieved source. If not, the graph loops back to draft
with a stricter prompt version. After two failed attempts, or when nothing relevant is found,
the ticket is escalated with a summary.
"""
import argparse
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .assistant import Assistant
from .llm import get_client

MAX_ATTEMPTS = 2
HIGH_WORDS = {"urgent", "lost", "stolen", "phishing", "suspicious", "outage", "crashed", "crash"}
LOW_PREFIXES = ("how can i", "how do i", "how to")


class State(TypedDict, total=False):
    question: str
    priority: str
    hits: list
    answer: str
    attempts: int
    prompt_version: str
    grounded: bool
    action: str
    ticket: dict
    trace: list


def classify_priority(question):
    q = question.lower()
    if any(w in q.replace("?", " ").replace(",", " ").split() for w in HIGH_WORDS):
        return "high"
    if q.startswith(LOW_PREFIXES):
        return "low"
    return "medium"


def build_graph(assistant):
    def classify(state):
        p = classify_priority(state["question"])
        return {"priority": p, "attempts": 0, "trace": ["classify:" + p]}

    def retrieve(state):
        hits = assistant.retriever.search(state["question"], 3)
        found = bool(hits) and hits[0].score >= assistant.min_score
        return {"hits": hits if found else [], "trace": state["trace"] + ["retrieve:" + ("found" if found else "none")]}

    def after_retrieve(state):
        return "draft" if state["hits"] else "escalate"

    def draft(state):
        version = "v1" if state["attempts"] == 0 else "v2"
        hits = state["hits"]
        context = "\n".join(f"[{h.id}] {h.title}: {h.text}" for h in hits)
        prompt = assistant.library.render("qa", version, context=context, question=state["question"])
        text = assistant.client.generate(prompt, question=state["question"], hits=hits)
        return {"answer": text, "attempts": state["attempts"] + 1, "prompt_version": version,
                "trace": state["trace"] + [f"draft:qa-{version}"]}

    def check(state):
        grounded = any(f"[{h.id}]" in state["answer"] for h in state["hits"])
        return {"grounded": grounded, "trace": state["trace"] + ["check:" + ("grounded" if grounded else "ungrounded")]}

    def after_check(state):
        if state["grounded"]:
            return "respond"
        return "draft" if state["attempts"] < MAX_ATTEMPTS else "escalate"

    def respond(state):
        return {"action": "answered", "trace": state["trace"] + ["respond"]}

    def escalate(state):
        question = state["question"]
        if assistant.client.name == "offline":
            summary = f"User reports: {question} No grounded answer was found. Needs L2 review."
        else:
            summary = assistant.client.generate(
                assistant.library.render("summarize", "v1", ticket=question), question=question, hits=[])
        ticket = {"priority": state["priority"], "question": question, "summary": summary, "level": "L2"}
        return {"action": "escalated", "ticket": ticket, "trace": state["trace"] + ["escalate"]}

    g = StateGraph(State)
    for name, fn in [("classify", classify), ("retrieve", retrieve), ("draft", draft),
                     ("check", check), ("respond", respond), ("escalate", escalate)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, {"draft": "draft", "escalate": "escalate"})
    g.add_edge("draft", "check")
    g.add_conditional_edges("check", after_check, {"respond": "respond", "draft": "draft", "escalate": "escalate"})
    g.add_edge("respond", END)
    g.add_edge("escalate", END)
    return g.compile()


def run(question, assistant=None):
    assistant = assistant or Assistant()
    return build_graph(assistant).invoke({"question": question})


def main():
    p = argparse.ArgumentParser(description="TicketFlow LangGraph support agent")
    p.add_argument("question", nargs="?")
    p.add_argument("--provider", choices=["offline", "groq"], default=None)
    p.add_argument("--graph", action="store_true", help="print the graph as a Mermaid diagram")
    args = p.parse_args()
    assistant = Assistant(client=get_client(args.provider))
    if args.graph:
        print(build_graph(assistant).get_graph().draw_mermaid())
        return
    if not args.question:
        p.error("question is required")
    out = build_graph(assistant).invoke({"question": args.question})
    print(f"[provider: {assistant.client.name}]")
    print("Path:", " -> ".join(out["trace"]))
    print("Priority:", out["priority"], "| Action:", out["action"])
    if out["action"] == "answered":
        print(out["answer"])
    else:
        t = out["ticket"]
        print(f"Ticket raised for {t['level']} | priority {t['priority']}\nSummary: {t['summary']}")


if __name__ == "__main__":
    main()
