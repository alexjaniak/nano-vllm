from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase


class ModelManager:
    # The model is Any: transformers 5 types generate() for the `ty` checker,
    # which pyright cannot bind, so no model annotation type-checks.
    def load_model(
        self, model_name: str, dtype_name: str = "auto"
    ) -> tuple[Any, PreTrainedTokenizerBase]:
        # "auto" loads the checkpoint's native precision (fp16/bf16 for most
        # modern models) instead of upcasting to fp32 — half the memory.
        # A dtype name like "float16" overrides it (e.g. GPUs without bf16).
        dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
