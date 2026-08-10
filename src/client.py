"""Interactive client: type a prompt, stream the completion back.

Usage: python client.py [--no-streaming]
"""

import argparse
import json
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def post(path: str, payload: dict[str, object]):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request)


def ask(prompt: str) -> None:
    with post("/generate", {"prompts": [prompt]}) as response:
        print(json.loads(response.read())["texts"][0])


def ask_streaming(prompt: str) -> None:
    with post("/generate_stream", {"prompt": prompt}) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            print(json.loads(data)["token"], end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive nano-vllm client")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="wait for the full completion instead of streaming tokens",
    )
    args = parser.parse_args()

    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        if args.no_streaming:
            ask(prompt)
        else:
            ask_streaming(prompt)


if __name__ == "__main__":
    main()
