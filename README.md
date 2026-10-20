# PromptLab

A Python assistant that answers IT help desk questions **only from a trusted knowledge base**, cites its source, and refuses or escalates when it cannot find an answer. It has two parts:

1. **PromptLab**: retrieval augmented generation (RAG) with a versioned prompt library
2. **TicketFlow**: a LangGraph agent that checks its own answer and raises a ticket when it cannot help

## Why this project

A plain LLM answers confidently even when it is wrong. PromptLab shows how to control that:
retrieve the right document first, give the model strict instructions through a versioned prompt, and refuse when nothing relevant is found.

## How it works

```
Question
   |
Retriever (text preprocessing + TF-IDF + cosine similarity, top 3)
   |
Score below threshold?  -- yes --> "I could not find this in the knowledge base."
   | no
Prompt library (qa v1 or qa v2) fills in context and question
   |
LLM client (Groq gpt-oss-120b, or offline fallback)
   |
Answer with source citations like [vpn-drops]
```

## Features

- **Prompt library**: named, versioned templates (qa v1, qa v2, summarize v1) with checks for missing variables
- **NLP retriever**: custom tokenizer with light stemming, TF-IDF with bigrams, cosine similarity
- **Out-of-scope guard**: refuses when the best retrieval score is too low
- **Switchable LLM**: Groq when `GROQ_API_KEY` is set, otherwise an offline extractive client that needs no internet
- **TicketFlow agent**: LangGraph state graph with conditional routing and a retry loop
- **Evaluation scripts** and **14 unit tests**

## Quick start

```
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m promptlab.cli "I forgot my password, how do I reset it?"
```

## Usage and example output

**Ask a question (offline mode, no API key needed)**
```
python -m promptlab.cli "VPN keeps dropping" --prompt v1
```
```
[provider: offline | prompt: qa:v1]
1. Frequent VPN drops are usually caused by weak home internet or a sleeping laptop. [vpn-drops]
2. Switch to a wired connection or move closer to the router, disable power saving for the network adapter, and reconnect to a different gateway. [vpn-drops]
Sources: vpn-drops, vpn-setup, recovery
```

**Unrelated question is refused**
```
python -m promptlab.cli "What is the capital of Australia?"
```
```
I could not find this in the knowledge base.
```

**Real LLM with Groq, comparing prompt versions**
```
set GROQ_API_KEY=your_key          (Windows CMD)
python -m promptlab.cli "VPN keeps dropping" --provider groq --prompt v2
```
```
[provider: groq | prompt: qa:v2]
1. Verify your home internet strength; a weak connection often causes drops. [vpn-drops]
2. Switch from Wi-Fi to a wired Ethernet connection or move closer to the router. [vpn-drops]
3. Disable power-saving options for your network adapter in the device's settings. [vpn-drops]
4. Reconnect to a different VPN gateway to see if the issue persists. [vpn-drops]
5. If drops continue, raise a ticket and include the exact times each drop occurs. [vpn-drops]
```

**Why prompt versions matter:** with the basic prompt (v1) the Groq model added generic steps that are not in the knowledge base, such as Device Manager menu paths and a checklist. The grounded prompt (v2) stayed inside the retrieved source and cited it on every step.

## TicketFlow: LangGraph support agent

```
classify -> retrieve -> draft -> check -> respond
                |                   |
                +----> escalate <---+
```

- **classify**: sets ticket priority (high, medium, low) with simple rules
- **retrieve**: finds relevant articles; goes straight to escalate if none are relevant
- **draft**: writes an answer using qa v1 first, qa v2 on a retry
- **check**: verifies the answer cites a retrieved source; if not, loops back to draft
- **escalate**: after two ungrounded attempts, raises an L2 ticket with a summary

**Self-correcting loop with a real LLM (Groq)**
```
python -m promptlab.agent "I lost my work laptop, what should I do?"
```
```
[provider: groq]
Path: classify:high -> retrieve:found -> draft:qa-v1 -> check:ungrounded -> draft:qa-v2 -> check:grounded -> respond
Priority: high | Action: answered
1. Report the lost laptop to the service desk and the security team right away. [lost-laptop]
2. Have the device locked remotely and request a reset of all your account passwords. [lost-laptop]
3. If the laptop was stolen, file a police report and include the reference number in your ticket. [lost-laptop]
```
The first draft had no citations, so the check node rejected it. The retry with the stricter prompt was grounded.

**Unrelated question is escalated (offline mode)**
```
python -m promptlab.agent "What is the capital of Australia?" --provider offline
```
```
Path: classify:medium -> retrieve:none -> escalate
Priority: medium | Action: escalated
Ticket raised for L2 | priority medium
```
Print the graph as a Mermaid diagram with `python -m promptlab.agent --graph`.

| Metric | Result |
|---|---|
| Correct article ranked first | 94.4% |
| Correct article in top 3 | 100% |
| Unrelated questions refused | 4 of 4 |
| TicketFlow answered help desk questions | 17 of 18 |
| TicketFlow escalated unrelated questions | 4 of 4 |

## Project structure

```
promptlab/
  prompts.py      versioned prompt library
  retriever.py    tokenizer, TF-IDF retriever
  llm.py          Groq client and offline client
  assistant.py    retrieval, prompt and LLM pipeline
  agent.py        TicketFlow LangGraph agent
  evaluate.py     retrieval evaluation
  agent_eval.py   agent evaluation
  cli.py          command line interface
data/             knowledge base (20 articles) and evaluation questions
tests/            14 unit tests
```

## Limitations

- The knowledge base is 20 synthetic help desk articles and the test set is small
- TF-IDF matches words, not meaning, so heavily reworded questions can miss (one such miss in the test set)
- Priority rules in TicketFlow are keyword based

## Tech

Python, scikit-learn, LangGraph, Groq API, unittest
