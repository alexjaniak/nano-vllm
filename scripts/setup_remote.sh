#!/usr/bin/env bash
# One-time setup on the rented GPU box. Assumes the instance was created from
# the vllm/vllm-openai image, so vLLM (and `vllm bench serve`) already exist
# system-wide; nano-vllm gets its own uv venv so the two torch pins can't
# fight each other.
#
#   bash scripts/setup_remote.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
REPO="${REPO:-https://github.com/alexjaniak/nano-vllm.git}"

echo "==> GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> Blackwell (sm_120) support"
# The single most likely way this whole benchmark silently goes wrong: a
# datacenter-only torch build that can't emit code for a consumer 5090.
python3 -c "
import torch
archs = torch.cuda.get_arch_list()
print('torch', torch.__version__, '| arch list:', archs)
assert any(a.startswith('sm_12') or a.startswith('sm_90') for a in archs), archs
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
"

echo "==> vLLM"
vllm --version

echo "==> uv + nano-vllm venv"
# The vllm/vllm-openai image is a slim runtime — no git, sometimes no curl.
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git curl; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d nano-vllm ] || git clone "$REPO"
cd nano-vllm && uv sync

echo "==> prefetch $MODEL (so model download isn't inside a timed phase)"
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
print('cached')
"

cat <<EOF

Setup done. From your laptop, open the tunnel so local Prometheus can scrape:

  ssh -N -L 8000:localhost:8000 -L 8001:localhost:8001 -p <VAST_PORT> root@<VAST_HOST>

Then locally:  cd observability && docker compose up -d
Then here:     bash scripts/sweep.sh
EOF
