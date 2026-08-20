# nano-vllm's GPU image. Standalone: a CUDA runtime base, uv, and this repo's
# lockfile — nothing from vLLM, nothing that knows bench/ exists.
#
# The CUDA minor matches the cuda-toolkit wheels torch pulls in uv.lock (13.0);
# torch ships its own CUDA libs, so the base is really just the driver ABI and
# a place to stand.
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

# Hugging Face downloads need a trust store; the CUDA base ships none.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Pinned: a build months from now resolves the same deps as today.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

# uv brings its own Python (.python-version pins 3.11; noble ships 3.12), so
# the image needs no distro python at all.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Deps from the lock alone, before src/: a code edit then rebuilds in seconds
# instead of re-resolving ~8GB of CUDA wheels.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY src/ /app/src/

# Fail the build rather than the first request.
RUN python -c "import torch, fastapi, transformers; print('torch', torch.__version__, 'cu' + str(torch.version.cuda))"

# Container defaults, overridable with -e or a trailing flag. --host lives here
# rather than in ENTRYPOINT: the 127.0.0.1 default is unreachable through -p,
# and an env var stays overridable without fighting argv precedence.
ENV NANO_HOST=0.0.0.0 \
    NANO_PORT=8001 \
    HF_HOME=/root/.cache/huggingface
EXPOSE 8001

# Readiness, not liveness — start-period covers the model load, which is
# minutes for an 8B on a cold HF cache. It reads NANO_PORT, so change the port
# with the env var rather than a --port flag and the check follows.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15m --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['NANO_PORT'] + '/ready')"

ENTRYPOINT ["python", "src/main.py"]
