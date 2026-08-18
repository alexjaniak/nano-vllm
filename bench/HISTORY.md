# Benchmark history

Every row is the same frozen spec on one GPU, both engines measured in the
same sweep. Only the engine changes. Rows are appended by `bench/report.py`.

| date | spec | nano-vllm | scenario | nano tok/s | vLLM tok/s | vLLM faster by | nano at | run |
|---|---|---|---|---|---|---|---|---|
| 2026-08-18 | v1 | `5fa1f7e` | fixed | 65.9 | 604.9 | 9.2× | 10.9% | `2026-08-18-qwen-qwen3-8b-v1-5fa1f7e` |
| 2026-08-18 | v1 | `5fa1f7e` | ragged | 64.4 | 640.4 | 10.0× | 10.0% | `2026-08-18-qwen-qwen3-8b-v1-5fa1f7e` |
| 2026-08-18 | v1 | `5fa1f7e` | ragged-sla | 64.2 | 630.3 | 9.8× | 10.2% | `2026-08-18-qwen-qwen3-8b-v1-5fa1f7e` |
| 2026-08-18 | v1 | `5fa1f7e` | prefix | 58.4 | 630.9 | 10.8× | 9.2% | `2026-08-18-qwen-qwen3-8b-v1-5fa1f7e` |
| 2026-08-18 | v1 | `5fa1f7e` | ragged-native | 65.0 | 2,951.6 | 45.4× | 2.2% | `2026-08-18-qwen-qwen3-8b-v1-5fa1f7e` |
