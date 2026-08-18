import torch
import torch.nn.functional as F


class SequenceCache:
    # One sequence's KV cache: a [seq_len, kv_heads, head_dim] tensor pair per
    # layer, grown by concatenation. Paged attention later replaces the
    # concat-on-append with block tables; the interface stays.
    def __init__(self, num_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers

    @property
    def seq_len(self) -> int:
        first = self.keys[0]
        return 0 if first is None else first.shape[0]

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Append this step's K/V and return the sequence's full cache.
        key, value = self.keys[layer], self.values[layer]
        self.keys[layer] = k if key is None else torch.cat([key, k])
        self.values[layer] = v if value is None else torch.cat([value, v])
        return self.keys[layer], self.values[layer]  # pyright: ignore[reportReturnType]


def varlen_attention(
    q: torch.Tensor,  # [total_q, heads, head_dim]
    k: torch.Tensor,  # [total_k, kv_heads, head_dim]
    v: torch.Tensor,  # [total_k, kv_heads, head_dim]
    cu_seqlens_q: torch.Tensor,  # [B+1] exclusive prefix sums
    cu_seqlens_k: torch.Tensor,
    scale: float,
) -> torch.Tensor:  # [total_q, heads, head_dim]
    # Attention over a packed ragged batch: sequences of any length share one
    # buffer, delimited by cu_seqlens. Causal masking is bottom-right aligned,
    # so a 1-token decode (q_len=1 vs k_len=L) sees its whole cache while a
    # prefill chunk stays causal — prefill and decode mix in the same batch.
    #
    # One sdpa call per sequence; a fused varlen kernel (paged attention)
    # replaces this whole function later. The dense layers upstream see the
    # packed buffer either way — only this loop is serial.
    groups = q.shape[1] // k.shape[1]  # GQA: query heads per KV head
    output = torch.empty_like(q)
    for i in range(cu_seqlens_q.numel() - 1):
        q_i = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].transpose(0, 1)  # [h, q, d]
        k_i = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)
        v_i = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)
        k_i = k_i.repeat_interleave(groups, dim=0)
        v_i = v_i.repeat_interleave(groups, dim=0)
        q_len, k_len = q_i.shape[1], k_i.shape[1]
        # tril offset by the cached prefix = bottom-right aligned causal.
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device).tril(
            diagonal=k_len - q_len
        )
        out = F.scaled_dot_product_attention(q_i, k_i, v_i, attn_mask=mask, scale=scale)
        output[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = out.transpose(0, 1)
    return output


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DecoderRunner:
    # The pre-norm decoder skeleton shared by the Llama family (Llama, Qwen,
    # Mistral, ...), driving the HF-loaded modules. The whole batch is one
    # packed [total_tokens, hidden] tensor — no batch dim, no padding.
    # Per-sequence raggedness exists only inside varlen_attention.
    #
    # Architecture quirks live in subclasses (models/*.py, one file per
    # family, dispatched by models/registry.py) — vLLM's layout in miniature.
    def __init__(self, model):
        config = model.config
        if "sliding_attention" in getattr(config, "layer_types", []):
            raise NotImplementedError("sliding-window layers not supported")
        self.model = model
        self.num_layers = config.num_hidden_layers
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

    def new_cache(self) -> SequenceCache:
        return SequenceCache(self.num_layers)

    def qkv(
        self, attn, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Project [total, hidden] -> per-head [total, heads, head_dim], before
        # RoPE. The default is plain projections;
        total = hidden.shape[0]
        q = attn.q_proj(hidden).view(total, -1, self.head_dim)
        k = attn.k_proj(hidden).view(total, -1, self.head_dim)
        v = attn.v_proj(hidden).view(total, -1, self.head_dim)
        return q, k, v

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,  # [total_tokens] packed across sequences
        position_ids: torch.Tensor,  # [total_tokens]
        q_lens: list[int],  # tokens per sequence: prompt_len (prefill) or 1
        caches: list[SequenceCache],
    ) -> torch.Tensor:  # [B, vocab] logits at each sequence's last token
        base = self.model.model  # the decoder stack inside the CausalLM wrapper
        total = input_ids.shape[0]
        device = input_ids.device

        # cumulative sequence lengths
        cu_q = torch.tensor([0] + q_lens, device=device).cumsum(0)
        k_lens = [cache.seq_len + q_len for cache, q_len in zip(caches, q_lens)]
        cu_k = torch.tensor([0] + k_lens, device=device).cumsum(0)

        hidden = base.embed_tokens(input_ids)

        # rotary_emb reads only dtype/device from its first arg.
        cos, sin = base.rotary_emb(hidden.unsqueeze(0), position_ids.unsqueeze(0))
        cos, sin = cos[0].unsqueeze(1), sin[0].unsqueeze(1)  # [total, 1, head_dim]

        for layer_idx, layer in enumerate(base.layers):
            attn = layer.self_attn
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            q, k, v = self.qkv(attn, hidden)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

            # Grow each sequence's cache, then pack past+new K/V ragged.
            packed_k, packed_v = [], []
            for i, cache in enumerate(caches):
                full_k, full_v = cache.append(
                    layer_idx, k[cu_q[i] : cu_q[i + 1]], v[cu_q[i] : cu_q[i + 1]]
                )
                packed_k.append(full_k)
                packed_v.append(full_v)

            out = varlen_attention(
                q, torch.cat(packed_k), torch.cat(packed_v), cu_q, cu_k, attn.scaling
            )
            hidden = residual + attn.o_proj(out.reshape(total, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        hidden = base.norm(hidden[cu_q[1:] - 1])  # last token of each sequence
        return self.model.lm_head(hidden)
