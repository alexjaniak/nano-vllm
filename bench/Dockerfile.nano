# nano-vllm, built ON TOP OF the exact vLLM image it gets benchmarked against.
#
# That is the point: identical torch, CUDA, driver and FlashAttention build on
# both sides, so a measured difference is nano-vllm's code and not its wheels.
# It also sidesteps the sm_120 wheel problem entirely — the base image already
# ships a Blackwell-capable torch.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.27.1
FROM ${VLLM_IMAGE}

# Constrain pip to the torch that is already here. Without this, resolving
# nano-vllm's deps can quietly swap in a different torch wheel and the whole
# same-toolchain argument evaporates — invisibly, as an unexplained kernel gap.
RUN python3 -c "import torch; print('torch==' + torch.__version__)" > /torch.pin \
 && cat /torch.pin

RUN pip install --no-cache-dir -c /torch.pin \
      "typer>=0.27.1" \
      "fastapi>=0.141.1" \
      "uvicorn>=0.52.1" \
      "prometheus-client>=0.26.0" \
      "transformers>=5.14.1"

# Fail the build rather than the experiment.
RUN test "$(python3 -c 'import torch; print(torch.__version__)')" = \
         "$(cut -d= -f3 /torch.pin)" \
 || { echo "FATAL: torch changed during dep install; the shared-base guarantee is broken"; exit 1; }

WORKDIR /app
COPY src/ /app/src/

EXPOSE 8001
ENTRYPOINT ["python3", "src/main.py"]
