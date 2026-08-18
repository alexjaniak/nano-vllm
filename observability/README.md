# Observability

Prometheus (:9090) scrapes both servers' `/metrics` every 1s; Grafana
(:3000, no login) serves the **nano-vllm vs vLLM** dashboard. Servers are
told apart by the scrape job label; a server that isn't running shows as
a `down` target.

> For the head-to-head benchmark use `bench/compose.yaml` instead — it brings
> up this same stack (plus `dcgm-exporter` for GPU counters) alongside the
> engines, scraping them by service name. The compose file here is for ad-hoc
> local poking at servers you started yourself.

```sh
docker compose up -d        # Prometheus + Grafana
```

Run the servers however you like; Prometheus expects each on its default
host port: nano-vllm on **8001**, vLLM on **8000**.

Benchmark both with the same load, one at a time:

```sh
docker run --rm --network host --entrypoint vllm vllm/vllm-openai:latest \
  bench serve --host 127.0.0.1 --port 8001 --model "$MODEL" \
  --dataset-name random --num-prompts 100 --request-rate 4
# repeat with --port 8000 for vLLM
```

Instrumentation lives in `src/llm/metrics.py` (vLLM's exact metric names)
and is driven from the scheduler loop in `src/llm/engine.py`. Panels for
features nano-vllm lacks (cache utilization, parallel sampling) stay empty
on its side by design.
