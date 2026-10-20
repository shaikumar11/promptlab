# PromptLab: RAG Assistant with a Versioned Prompt Library

Python app that answers IT help desk questions from a knowledge base using retrieval augmented generation.

## What it has
- Prompt library: named, versioned templates (qa v1, qa v2, summarize v1) with variable checks
- NLP retriever: custom tokenizer with light stemming, TF-IDF with bigrams, cosine similarity, top-k
- Switchable LLM client: Groq Llama 3.3 70B when GROQ_API_KEY is set, offline extractive client otherwise
- Out-of-scope guard: refuses when the best retrieval score is below a threshold
- Evaluation script: hit at 1, hit at 3, false reject rate, out-of-scope rejection
- 9 unit tests

## Run
    pip install scikit-learn
    python -m promptlab.cli "I forgot my password, how do I reset it?"
    python -m promptlab.cli "VPN keeps dropping" --prompt v1
    python -m promptlab.evaluate
    python -m unittest discover -s tests -v

    # real LLM answers
    set GROQ_API_KEY=your_key        (Windows)
    python -m promptlab.cli "VPN keeps dropping" --provider groq

## Results on the included eval set (18 in scope, 4 out of scope, synthetic data)
hit at 1: 94.4 percent, hit at 3: 100 percent, out-of-scope rejected: 100 percent, false reject: 5.6 percent
