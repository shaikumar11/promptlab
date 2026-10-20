"""Measures retrieval quality and out-of-scope rejection on a small labelled set."""
import json
import os
import sys
from .assistant import Assistant, DATA
from .llm import OfflineClient


def evaluate(min_score=0.12, verbose=False):
    with open(os.path.join(DATA, "eval_set.json"), encoding="utf-8") as f:
        data = json.load(f)
    a = Assistant(client=OfflineClient(), min_score=min_score)
    hit1 = hit3 = false_reject = 0
    for item in data["in_scope"]:
        hits = a.retriever.search(item["q"], 3)
        ids = [h.id for h in hits]
        hit1 += bool(ids) and ids[0] == item["doc"]
        hit3 += item["doc"] in ids
        false_reject += a.ask(item["q"]).refused
        if verbose and (not ids or ids[0] != item["doc"]):
            print("MISS:", item["q"], "->", [(h.id, round(h.score, 2)) for h in hits])
    rejected = sum(a.ask(q).refused for q in data["out_of_scope"])
    if verbose:
        for q in data["out_of_scope"]:
            h = a.retriever.search(q, 1)
            print("OUT:", q, "->", [(x.id, round(x.score, 2)) for x in h])
    n, m = len(data["in_scope"]), len(data["out_of_scope"])
    return {"in_scope": n, "out_of_scope": m,
            "hit_at_1": round(hit1 / n, 3), "hit_at_3": round(hit3 / n, 3),
            "false_reject_rate": round(false_reject / n, 3),
            "out_of_scope_rejected": round(rejected / m, 3)}


if __name__ == "__main__":
    ms = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12
    print(json.dumps(evaluate(ms, verbose=True), indent=2))
