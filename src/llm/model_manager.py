import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase


class ModelManager:
    # The model is Any: transformers 5 types generate() for the `ty` checker,
    # which pyright cannot bind, so no model annotation type-checks.
    def load_model(self, model_name: str) -> tuple[Any, PreTrainedTokenizerBase]:
        # "auto" loads the checkpoint's native precision (fp16/bf16 for most
        # modern models) instead of upcasting to fp32 — half the memory.
        # NANO_VLLM_DTYPE overrides it (e.g. float16 on GPUs without bf16).
        dtype_name = os.environ.get("NANO_VLLM_DTYPE", "auto")
        dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
