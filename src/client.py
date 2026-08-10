"""Quick interactive client: type a prompt, get a completion back.

Usage: python client.py  (with the server running on port 8000)
"""

import json
import urllib.request

URL = "http://127.0.0.1:8000/generate"


def ask(prompt: str) -> str:
    body = json.dumps({"prompts": [prompt]}).encode()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["texts"][0]


if __name__ == "__main__":
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        print(ask(prompt))
