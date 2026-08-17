from ..model_runner import DecoderRunner


class Qwen3Runner(DecoderRunner):
    # Qwen3 is the Llama-family skeleton plus RMSNorm on each head's q and k
    # (applied over head_dim, before RoPE). Everything else is inherited.
    def qkv(self, attn, hidden):
        total = hidden.shape[0]
        q = attn.q_norm(attn.q_proj(hidden).view(total, -1, self.head_dim))
        k = attn.k_norm(attn.k_proj(hidden).view(total, -1, self.head_dim))
        v = attn.v_proj(hidden).view(total, -1, self.head_dim)
        return q, k, v
