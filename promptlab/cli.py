import argparse
from .assistant import Assistant
from .llm import get_client


def main():
    p = argparse.ArgumentParser(description="Ask the help desk assistant a question")
    p.add_argument("question")
    p.add_argument("--prompt", default="v2", help="qa prompt version, v1 or v2")
    p.add_argument("--provider", choices=["offline", "groq"], default=None)
    args = p.parse_args()
    client = get_client(args.provider)
    ans = Assistant(client=client).ask(args.question, args.prompt)
    print(f"[provider: {client.name} | prompt: qa:{args.prompt}]")
    print(ans.text)
    if ans.sources:
        print("Sources:", ", ".join(ans.sources))


if __name__ == "__main__":
    main()
