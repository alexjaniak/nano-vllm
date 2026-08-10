from typing import Any

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase


class ModelManager:
    # The model is Any: transformers 5 types generate() for the `ty` checker,
    # which pyright cannot bind, so no model annotation type-checks.
    def load_model(self, model_name: str) -> tuple[Any, PreTrainedTokenizerBase]:
        # dtype="auto" loads the checkpoint's native precision (fp16/bf16 for
        # most modern models) instead of upcasting to fp32 — half the memory.
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
