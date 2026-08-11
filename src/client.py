"""Interactive client: type a prompt, stream the completion back.

Usage: python client.py [--no-streaming]
"""

import argparse
import json
import urllib.request

def get_model(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/v1/models") as response:
        return json.load(response)["data"][0]["id"]


def complete(base_url: str, model: str, prompt: str, temperature: float, max_tokens: int, stream: bool):
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request)


def ask(base_url: str, model: str, prompt: str, temperature: float, max_tokens: int) -> None:
    with complete(base_url, model, prompt, temperature, max_tokens, stream=False) as response:
        print(json.loads(response.read())["choices"][0]["text"])


def ask_streaming(base_url: str, model: str, prompt: str, temperature: float, max_tokens: int) -> None:
    with complete(base_url, model, prompt, temperature, max_tokens, stream=True) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            print(json.loads(data)["choices"][0]["text"], end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive nano-vllm client")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="wait for the full completion instead of streaming tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling randomness; 0 for greedy decoding",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="cap on generated tokens per completion",
    )
    args = parser.parse_args()

    model = get_model(args.url)
    print(f"model: {model}")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        if args.no_streaming:
            ask(args.url, model, prompt, args.temperature, args.max_tokens)
        else:
            ask_streaming(args.url, model, prompt, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
