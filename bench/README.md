# bench — head-to-head against vLLM, on a frozen spec

The benchmark is infrastructure, not a thing to iterate on. `spec-v1.toml` is
frozen: the same workload, the same model commit, the same pinned vLLM, every
time. Only the engine changes. That is what makes a run from next year
comparable to a run from today.

Improving the benchmark means writing `spec-v2.toml` and taking a fresh
baseline for both engines. It does **not** mean editing v1 — every prior run
would silently stop meaning what it says. `run.py` hashes the spec into each
`manifest.json` so that drift is detectable rather than merely discouraged.

## What the host has to provide

Everything heavy is in containers, so the host needs almost nothing. But it
does need these five, and `run.py --dry-run` will not catch them — the real
preflight runs at the start of a sweep and fails on each with a specific error.

| requirement | why |
|---|---|
| NVIDIA driver + `nvidia-smi` | the sweep polls it to confirm VRAM is actually released between engines |
| **open** kernel modules on consumer Blackwell | a 5090 (`sm_120`) does not work with the proprietary flavor — see below |
| **its own Docker daemon** | see below — this is the one that bites |
| NVIDIA container toolkit | without it containers start CPU-only and the sweep silently measures nothing |
| Docker Compose v2 | `compose.yaml` uses the v2 `deploy.resources` GPU syntax |
| Python 3.11+ | `run.py` is stdlib-only but uses `tomllib`. Ubuntu 22.04 ships 3.10; 24.04 is fine |

`bench/provision.sh` fixes everything on that list except the driver, which
needs a reboot, and reports precisely what to do when it hits one:

```sh
sudo bash bench/provision.sh
```

~80GB of disk: the pinned vLLM image is ~20GB, Qwen3-8B ~16GB, plus the
nano-vllm layer on top.

### The Docker-inside-Docker catch

Most rented GPU "instances" are themselves containers — that is how the
marketplace isolates you. You cannot run this rig inside one, because it needs
to start containers of its own.

So pick a host that gives you a real machine: a cloud GPU VM (GCP, AWS, Azure,
Lambda, Crusoe), bare metal, or — on marketplaces like Vast.ai — specifically a
**VM instance** rather than a standard one. Vast's VM templates use KVM with
GPU passthrough and ship CUDA and Docker already; standard Vast instances are
containers and won't work.

Verify in one command before renting anything for an hour:

```sh
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L
```

If that prints your GPU, the host is good. If it can't reach a daemon, the box
is a container. If it runs but sees no GPU, the container toolkit isn't wired
in (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

### The Blackwell open-modules catch

A 5090 requires the **open** NVIDIA kernel modules. Given the proprietary
flavor of the same driver version, the module loads, initializes, and then
finds no devices — `nvidia-smi` reports "No devices were found" while `lspci`
happily shows the card, so it reads like broken passthrough rather than a
wrong package. `dmesg` names it outright:

```
NVRM: The NVIDIA GPU 0000:00:07.0 (PCI ID: 10de:2b85) installed in this
NVRM: system requires use of the NVIDIA open kernel modules.
```

One command tells you which flavor is installed — `Dual MIT/GPL` is open,
`NVIDIA` is not:

```sh
modinfo nvidia | grep -i license
```

Images built for datacenter fleets (A100/H100) ship the proprietary modules,
which is correct for those cards and wrong for this one.

### Pin the Vast image tag

Vast VM templates can only use images from `docker.io/vastai/kvm` — a custom
image will not launch, so the tag is the only lever. Leave **VM Version** on
`@vastai-automatic-tag` and Vast picks per rental, which is how a jammy guest
turns up under a template you believed was noble. Pin an explicit tag instead,
for the same reason `spec-v1.toml` pins a model commit rather than `main`.

Choose a CLI/terminal tag, never a desktop one: a desktop guest runs a display
server holding VRAM, and `run.py` refuses to start when more than 2 GB is
already in use. CUDA 12.8 is the floor — `sm_120` does not exist below it.

## Run it

```sh
git clone <repo> && cd nano-vllm
python3 bench/run.py --dry-run     # the plan and the pins, touches nothing
python3 bench/run.py               # ~95 min at v0 speeds; shrinks as nano-vllm improves
python3 bench/report.py            # Markdown table + one row appended to HISTORY.md
```

No venv, no `uv`, no deps to install — `run.py` and `report.py` are stdlib
only. Only `chart.py` needs matplotlib, and it reads committed result files, so
run it on your laptop afterwards instead:

```sh
scp -r <host>:nano-vllm/experiments/<run> ./experiments/
uv run --with matplotlib bench/chart.py
```

Useful flags: `--only nano-vllm` and `--scenario ragged` resume a sweep that
died partway; `--no-obs` skips Prometheus/Grafana.

To watch Grafana while it runs: `ssh -N -L 3000:localhost:3000 <host>`.

## The metric

**nano-vllm output tok/s ÷ vLLM output tok/s**, both measured in the same
sweep on the same box within the same hour.

A ratio is self-normalizing. Absolute throughput moves when you rent a
different 5090, when the driver bumps, when the box runs hot — the ratio
doesn't. `HISTORY.md` tracks it per scenario per run, and that table is the
actual deliverable.

## Scenarios

| scenario | shape | why it exists |
|---|---|---|
| `fixed` | 512 in / 128 out | bridge to the v0 baseline (2026-08-13); mostly a kernel-speed signal |
| `ragged` | 512 in / 512±75% out | **the headline** — variable output lengths are where scheduling, head-of-line blocking and KV fragmentation show up |
| `ragged-sla` | same, rate 4, goodput SLA | reads ~0% until the engine keeps up, then climbs — stays meaningful after raw tok/s stops discriminating |
| `prefix` | 2k shared prefix | worth nothing without prefix caching; frozen in now so the feature has a before-number |
| `ragged-native` | same load, each engine at its OWN `max_num_seqs` | the deployment number, and where a future paged-KV concurrency win can appear |

The first four hold both engines at `max_num_seqs=8` — v0's cap, which is what
isolates executor quality from "vLLM just batches wider". `ragged-native` drops
that constraint deliberately.

Sizing targets the *slow* engine. Runs get shorter as nano-vllm improves, which
is the right direction for a spec that never changes.

## Why containers

Three things the old `sweep.sh` couldn't do:

- **`docker stop` is a hard VRAM guarantee.** The old script needed
  `supervisorctl stop` + `pkill -f "vllm serve"` + `sleep 5` to wrestle the GPU
  away from the host image's autostarted vLLM, then hoped. `run.py` polls
  `nvidia-smi` until memory actually comes back before starting the next engine.
- **The measuring instrument is pinned.** `vllm bench serve` runs from the same
  pinned image as the vLLM under test. Previously it was whatever the host
  image shipped, and its metric definitions have moved across releases.
- **Any commit can be re-baselined.** `docker/Dockerfile` builds from the repo
  and its lockfile, so checking out an old SHA and re-running reproduces that
  commit's numbers — same deps, same frozen opponent.

Each engine runs the image it ships as: vLLM its pinned release image,
nano-vllm the standalone `docker/Dockerfile` built from `uv.lock`. That is the
honest comparison — what someone would actually deploy — but it means the two
sides no longer share wheels. `run.py` reads torch/CUDA from inside both
containers into `manifest.json` and warns when they differ; a kernel-level
claim has to argue past that difference rather than assume it away.

## GPU metrics

`dcgm-exporter` is scraped alongside both engines. The metric that carries the
batching argument is **`DCGM_FI_PROF_DRAM_ACTIVE`**: single-sequence decode is
memory-bandwidth bound, because the entire model is read to produce one token.
A working batched forward pass therefore shows up as bandwidth staying pinned
while tok/s multiplies — bytes-per-token collapsing. `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`
is the other half of that story, and `DCGM_FI_DEV_FB_USED` is where paged KV
will eventually show up.

Caveat worth checking on the box before trusting it: the `DCGM_FI_PROF_*`
profiling counters are historically restricted on consumer GeForce cards, and
whether a 5090 exposes them is unverified. `FB_USED`, `GPU_UTIL`, power and
clocks always work. If the profiling counters are missing, effective bandwidth
can be derived instead — model bytes × steps/s — which is less direct but makes
the same point.

```sh
curl -s localhost:9400/metrics | grep -E 'DRAM_ACTIVE|TENSOR_ACTIVE|FB_USED'
```

## Provenance

Every run writes `manifest.json`: nano-vllm git SHA and dirty flag, vLLM image
digest, torch/CUDA/arch list read from inside both containers, GPU name and
driver, the pinned model commit, and the spec hash. A run whose `src/` was
dirty is flagged loudly — it cannot be reproduced from a commit.
