"""Interactive client: type a prompt, stream the completion back."""

import json
import urllib.request
from typing import Annotated

import typer

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


cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    url: Annotated[
        str, typer.Option(help="Base URL of the nano-vllm server.")
    ] = "http://127.0.0.1:8000",
    streaming: Annotated[
        bool,
        typer.Option(help="Stream tokens as they generate; --no-streaming waits for the full completion."),
    ] = True,
    temperature: Annotated[
        float, typer.Option(help="Sampling randomness; 0 for greedy decoding.")
    ] = 0.7,
    max_tokens: Annotated[
        int, typer.Option(help="Cap on generated tokens per completion.")
    ] = 100,
) -> None:
    """Interactive nano-vllm client: type a prompt, get the completion back."""
    model = get_model(url)
    print(f"model: {model}")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        if streaming:
            ask_streaming(url, model, prompt, temperature, max_tokens)
        else:
            ask(url, model, prompt, temperature, max_tokens)


if __name__ == "__main__":
    cli()
