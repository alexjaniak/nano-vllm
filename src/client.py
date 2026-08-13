"""Interactive client: type a prompt, stream the completion back."""

import json
import urllib.request
from typing import Annotated

import typer

# Shares the server's sampling defaults (torch-free import) so the two
# sides can't drift apart.
from llm import SamplingParams

_DEFAULTS = SamplingParams()


def get_model(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/v1/models") as response:
        return json.load(response)["data"][0]["id"]


def complete(base_url: str, payload: dict):
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request)


def ask(base_url: str, payload: dict) -> None:
    with complete(base_url, payload | {"stream": False}) as response:
        print(json.loads(response.read())["choices"][0]["text"])


def ask_streaming(base_url: str, payload: dict) -> None:
    with complete(base_url, payload | {"stream": True}) as response:
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
    ] = "http://127.0.0.1:8001",
    streaming: Annotated[
        bool,
        typer.Option(help="Stream tokens as they generate; --no-streaming waits for the full completion."),
    ] = True,
    temperature: Annotated[
        float, typer.Option(help="Sampling randomness; 0 for greedy decoding.")
    ] = _DEFAULTS.temperature,
    top_p: Annotated[
        float, typer.Option(help="Nucleus sampling probability mass; 1.0 disables.")
    ] = _DEFAULTS.top_p,
    top_k: Annotated[
        int, typer.Option(help="Sample only among the k most likely tokens; -1 disables.")
    ] = _DEFAULTS.top_k,
    max_tokens: Annotated[
        int, typer.Option(help="Cap on generated tokens per completion.")
    ] = _DEFAULTS.max_tokens,
    stop: Annotated[
        list[str] | None,
        typer.Option(help="Stop string; repeat the flag for multiple."),
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="RNG seed for reproducible sampling.")
    ] = None,
) -> None:
    """Interactive nano-vllm client: type a prompt, get the completion back."""
    model = get_model(url)
    print(f"model: {model}")
    payload = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stop": stop or None,  # typer gives [] when the flag is absent
        "seed": seed,
    }
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        if streaming:
            ask_streaming(url, payload | {"prompt": prompt})
        else:
            ask(url, payload | {"prompt": prompt})


if __name__ == "__main__":
    cli()
