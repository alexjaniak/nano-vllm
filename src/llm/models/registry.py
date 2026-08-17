# Architecture registry, after vLLM's ModelRegistry: HF checkpoints
# self-identify via config.architectures, and unknown ones fail loudly —
# a wrong-but-similar architecture would emit garbage, not errors.
from ..model_runner import DecoderRunner
from .qwen3 import Qwen3Runner

ARCHITECTURES: dict[str, type[DecoderRunner]] = {
    "Qwen3ForCausalLM": Qwen3Runner,
}


def get_runner(model) -> DecoderRunner:
    arch = (model.config.architectures or ["?"])[0]
    runner_cls = ARCHITECTURES.get(arch)
    if runner_cls is None:
        raise ValueError(
            f"unsupported architecture {arch!r}; supported: {sorted(ARCHITECTURES)}"
        )
    return runner_cls(model)
