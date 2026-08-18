#!/usr/bin/env python3
"""Frozen-spec head-to-head sweep: vLLM vs nano-vllm, one GPU, one engine up
at a time.

    python3 bench/run.py                      # full sweep against spec-v1.toml
    python3 bench/run.py --dry-run            # print the plan + manifest, touch nothing
    python3 bench/run.py --only nano-vllm     # one engine (resume a half-finished sweep)
    python3 bench/run.py --scenario ragged    # one scenario

Stdlib only, so it runs on a bare rented GPU box with no venv.

The spec is frozen; this script is not. Anything that changes what is measured
belongs in spec-v*.toml, anything that changes how it is driven belongs here.
"""

from __future__ import annotations

import sys

# Ubuntu 22.04 — the usual rented-box image — still ships 3.10, and tomllib
# landed in 3.11. Say so plainly instead of dying on an ImportError.
if sys.version_info < (3, 11):
    raise SystemExit(
        f"bench/run.py needs Python 3.11+ for tomllib; this is {sys.version.split()[0]}.\n"
        f"It is the only thing this script needs beyond the stdlib — "
        f"`apt install python3.11` and run it with that."
    )

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
COMPOSE = ["docker", "compose", "-f", str(BENCH_DIR / "compose.yaml")]

# Enough headroom for a model to have fully released; a wedged container would
# otherwise poison the next engine's run and look like a regression.
VRAM_IDLE_MIB = 2048


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def sh_ok(cmd: list[str], **kw) -> str:
    proc = sh(cmd, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}… failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def gpu_mem_used_mib() -> int:
    out = sh_ok(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return max(int(line) for line in out.splitlines())


def wait_vram_free(timeout: int = 180) -> None:
    """Container stop is not instant, and the driver frees a process's
    allocations only when it actually dies."""
    for _ in range(timeout):
        used = gpu_mem_used_mib()
        if used < VRAM_IDLE_MIB:
            log(f"VRAM released ({used} MiB in use)")
            return
        time.sleep(1)
    log(f"WARNING: {gpu_mem_used_mib()} MiB still in use after {timeout}s")


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


def compose(args: list[str], env: dict[str, str], profile: str | None = None,
            check: bool = True, **kw) -> subprocess.CompletedProcess:
    cmd = COMPOSE + (["--profile", profile] if profile else []) + args
    proc = subprocess.run(cmd, text=True, env={**os.environ, **env}, **kw)
    if check and proc.returncode != 0:
        raise RuntimeError(f"compose {' '.join(args[:2])} failed ({proc.returncode})")
    return proc


def wait_healthy(port: int, path: str, timeout: int = 1200) -> None:
    """Model load is minutes, and on a cold HF cache can be much longer."""
    url = f"http://127.0.0.1:{port}{path}"
    log(f"waiting for {url}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    log("healthy")
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(2)
    raise RuntimeError(f"{url} never became healthy after {timeout}s")


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def flag_args(args: dict) -> list[str]:
    """{"random-input-len": 512, "goodput": ["ttft:2000"]} -> CLI flags."""
    out: list[str] = []
    for key, value in args.items():
        out.append(f"--{key}")
        out.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return out


def base_env(spec: dict, results_dir: Path, max_num_seqs: int) -> dict[str, str]:
    model, run = spec["model"], spec["run"]
    return {
        "VLLM_IMAGE": spec["engines"]["vllm"]["image"],
        "MODEL": model["id"],
        "MODEL_REVISION": model["revision"],
        "DTYPE": model["dtype"],
        "MAX_MODEL_LEN": str(model["max_model_len"]),
        "GPU_MEM_UTIL": str(run["gpu_memory_utilization"]),
        "SEED": str(run["seed"]),
        "MAX_NUM_SEQS": str(max_num_seqs),
        "RESULTS_DIR": str(results_dir),
        "HF_CACHE": os.environ.get("HF_CACHE", str(Path.home() / ".cache" / "huggingface")),
        "VLLM_PORT": str(spec["engines"]["vllm"]["port"]),
        "NANO_PORT": str(spec["engines"]["nano-vllm"]["port"]),
    }


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def git(*args: str) -> str:
    return sh_ok(["git", "-C", str(REPO), *args])


def image_digest(image: str) -> str:
    raw = sh_ok(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image])
    digests = json.loads(raw)
    return digests[0] if digests else "<local build, no digest>"


def toolchain(image: str, env: dict[str, str]) -> dict:
    """Read torch/CUDA from inside an image — the versions that actually ran,
    not whatever the host happens to have."""
    code = (
        "import torch, json;"
        "print(json.dumps({'torch': torch.__version__,"
        "'cuda': torch.version.cuda,"
        "'arch_list': torch.cuda.get_arch_list()}))"
    )
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", code],
        text=True, capture_output=True, env={**os.environ, **env},
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-400:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def build_manifest(spec: dict, spec_path: Path, env: dict[str, str]) -> dict:
    spec_bytes = spec_path.read_bytes()
    dirty = bool(git("status", "--porcelain", "--", "src"))
    gpu = sh_ok(["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"]).splitlines()[0]
    name, mem, driver = (part.strip() for part in gpu.split(","))
    return {
        "spec_version": spec["version"],
        "spec_file": spec_path.name,
        # The hash is the point: it proves months later that the spec did not
        # drift between two runs being compared.
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nano_vllm": {
            "git_sha": git("rev-parse", "HEAD"),
            "git_describe": git("describe", "--always", "--dirty", "--tags"),
            "dirty": dirty,
            "subject": git("log", "-1", "--format=%s"),
        },
        "vllm": {
            "image": spec["engines"]["vllm"]["image"],
            "digest": image_digest(spec["engines"]["vllm"]["image"]),
        },
        "model": dict(spec["model"]),
        "gpu": {"name": name, "memory_total": mem, "driver": driver},
        "toolchain": {
            "vllm": toolchain(spec["engines"]["vllm"]["image"], env),
            "nano-vllm": toolchain("nano-vllm:bench", env),
        },
    }


def assert_same_toolchain(manifest: dict) -> None:
    """The shared-base guarantee, verified rather than assumed. If these
    diverge, every kernel-level claim in the writeup is unfounded."""
    a = manifest["toolchain"]["vllm"]
    b = manifest["toolchain"]["nano-vllm"]
    if "error" in a or "error" in b:
        log(f"WARNING: could not read toolchain ({a.get('error') or b.get('error')})")
        return
    if (a["torch"], a["cuda"]) != (b["torch"], b["cuda"]):
        raise SystemExit(
            f"FATAL: engines are on different toolchains — "
            f"vllm torch {a['torch']}/cu{a['cuda']} vs "
            f"nano torch {b['torch']}/cu{b['cuda']}. "
            f"Rebuild nano-vllm:bench from the pinned VLLM_IMAGE."
        )
    log(f"toolchain matched: torch {a['torch']}, cuda {a['cuda']}")


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def preflight(spec: dict, env: dict[str, str], scenarios: list[dict]) -> None:
    """Everything the host must provide, checked every run rather than assumed.
    Each of these has a specific way of ruining a sweep 40 minutes in."""
    for tool in ("docker", "nvidia-smi"):
        if not shutil.which(tool):
            raise SystemExit(f"FATAL: {tool} not found on PATH")

    if sh(["docker", "info"]).returncode != 0:
        raise SystemExit(
            "FATAL: the docker daemon is unreachable. If this host is itself a container "
            "(most rented GPU 'instances' are), you need a VM or bare-metal box instead — "
            "this rig needs its own docker daemon."
        )
    if sh(["docker", "compose", "version"]).returncode != 0:
        raise SystemExit("FATAL: docker compose v2 plugin missing")

    # The quietest failure mode there is: docker works, the GPU works, but the
    # NVIDIA container toolkit was never wired into the daemon — so every engine
    # container starts CPU-only and the whole sweep measures nothing.
    if sh(["docker", "run", "--rm", "--gpus", "all", "ubuntu:22.04", "nvidia-smi", "-L"]).returncode != 0:
        raise SystemExit(
            "FATAL: containers cannot see the GPU. Install the NVIDIA container toolkit, then:\n"
            "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        )

    # Pinned vLLM image ~20GB + Qwen3-8B ~16GB + the nano-vllm layer. Rented
    # boxes default to a small disk and this is the classic mid-sweep death.
    free_gb = shutil.disk_usage(REPO).free // 2**30
    if free_gb < 80:
        raise SystemExit(f"FATAL: {free_gb}GB free on {REPO}, need ~80GB for images + weights")

    used = gpu_mem_used_mib()
    if used > VRAM_IDLE_MIB:
        raise SystemExit(
            f"FATAL: {used} MiB of VRAM already in use. Something else owns the GPU "
            f"(a leftover container, or an autostarted server from the host image). "
            f"Clear it first — a benchmark sharing a GPU measures nothing."
        )
    log(f"host ok: docker + container GPU access, {free_gb}GB free, GPU idle")

    log(f"pulling {spec['engines']['vllm']['image']}")
    sh_ok(["docker", "pull", spec["engines"]["vllm"]["image"]])

    log("building nano-vllm:bench")
    compose(["--profile", "nano-vllm", "build", "nano-vllm"], env)

    # Fail on an unknown bench flag now, not 40 minutes into a sweep. The
    # random-dataset flags have moved across vLLM releases; the image is pinned
    # so this is a one-time check, but it must actually happen.
    log("checking harness flags")
    # Help may land on stderr, and argparse wraps long option names at the
    # terminal width — an 80-column default splits --random-input-len across
    # lines and every flag reads as missing. Widen, take both streams, unwrap.
    proc = sh(["docker", "run", "--rm", "-e", "COLUMNS=200", "--entrypoint", "vllm",
               spec["engines"]["vllm"]["image"], "bench", "serve", "--help"])
    helptext = re.sub(r"-\s*\n\s*", "-", proc.stdout + proc.stderr)
    wanted = {f"--{k}" for s in scenarios for k in s["args"]} | {"--request-rate", "--num-prompts"}
    missing = sorted(f for f in wanted if f not in helptext)
    if missing and "--num-prompts" in helptext:
        # Help was readable and the flag genuinely is not there.
        raise SystemExit(
            f"FATAL: `vllm bench serve` in {spec['engines']['vllm']['image']} does not "
            f"accept {', '.join(missing)}. The spec and the pinned harness disagree."
        )
    if missing:
        # Could not read the help at all; do not block the sweep on the checker.
        # A bad flag then surfaces at the first scenario, not 40 minutes in.
        log(f"WARNING: could not parse `vllm bench serve --help` "
            f"({len(helptext)} bytes, exit {proc.returncode}) — flag check skipped")

    # Prefetch weights so no download lands inside a timed phase.
    log(f"prefetching {spec['model']['id']}@{spec['model']['revision'][:12]}")
    code = (
        "from huggingface_hub import snapshot_download;"
        f"snapshot_download('{spec['model']['id']}', revision='{spec['model']['revision']}',"
        "allow_patterns=['*.json','*.safetensors','*.txt','*.model'])"
    )
    compose(["--profile", "bench", "run", "--rm", "--entrypoint", "python3",
             "bench", "-c", code], env)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


class Timeline:
    def __init__(self, path: Path, grafana: str):
        self.path, self.grafana = path, grafana
        path.write_text("engine,phase,scenario,start_ms,end_ms,duration_s,grafana_url\n")

    def record(self, engine: str, phase: str, scenario: str, t0: int, t1: int, pad_s: int) -> None:
        url = (f"{self.grafana}/d/nano-vllm-vs-vllm?from={t0 - pad_s * 1000}"
               f"&to={t1 + pad_s * 1000}&var-job={engine}")
        with self.path.open("a") as fh:
            fh.write(f"{engine},{phase},{scenario},{t0},{t1},{(t1 - t0) // 1000},{url}\n")


def run_scenario(engine: str, scenario: dict, spec: dict, env: dict[str, str],
                 out: Path, timeline: Timeline) -> None:
    gap = spec["run"]["quiet_gap_s"]
    name, port = scenario["name"], spec["engines"][engine]["port"]

    log(f"quiet {gap}s")
    time.sleep(gap)

    args = [
        "--backend", "openai",
        "--base-url", f"http://{engine}:{port}",
        "--model", spec["model"]["id"],
        "--num-prompts", str(scenario["num_prompts"]),
        "--request-rate", str(scenario["request_rate"]),
        "--seed", str(spec["run"]["seed"]),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-dir", "/results",
        "--result-filename", f"{engine}__{name}.json",
        *flag_args(scenario["args"]),
    ]

    t0 = now_ms()
    log(f"=== {engine} / {name}  (rate={scenario['request_rate']}, n={scenario['num_prompts']})")
    proc = compose(["--profile", "bench", "run", "--rm", "bench", *args], env,
                   check=False, capture_output=True, text=True)
    t1 = now_ms()

    (out / f"{engine}__{name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        log(f"WARNING: {engine}/{name} exited {proc.returncode} — see the log")
    else:
        log(f"=== {engine} / {name} done in {(t1 - t0) // 1000}s")
    timeline.record(engine, "measured", name, t0, t1, gap)


def sweep_engine(engine: str, spec: dict, scenarios: list[dict], results_dir: Path,
                 out: Path, timeline: Timeline) -> None:
    cfg = spec["engines"][engine]
    matched = spec["run"]["matched_max_num_seqs"]

    # Group by config so the engine restarts once per max_num_seqs, not once
    # per scenario.
    for config in ("matched", "native"):
        todo = [s for s in scenarios if s["config"] == config]
        if not todo:
            continue
        cap = matched if config == "matched" else cfg["native_max_num_seqs"]
        env = base_env(spec, results_dir, cap)

        log(f"--- {engine}: {config} config, max_num_seqs={cap}")
        compose(["--profile", engine, "up", "-d", engine], env)
        try:
            wait_healthy(cfg["port"], cfg["health_path"])

            w0 = now_ms()
            log(f"warmup ({engine})")
            compose(["--profile", "bench", "run", "--rm", "bench",
                     "--backend", "openai",
                     "--base-url", f"http://{engine}:{cfg['port']}",
                     "--model", spec["model"]["id"],
                     "--dataset-name", "random",
                     "--random-input-len", "512", "--random-output-len", "32",
                     "--num-prompts", str(spec["run"]["warmup_prompts"]),
                     "--request-rate", "4", "--seed", str(spec["run"]["seed"])],
                    env, check=False, capture_output=True)
            timeline.record(engine, "warmup", config, w0, now_ms(), spec["run"]["quiet_gap_s"])

            for scenario in todo:
                run_scenario(engine, scenario, spec, env, out, timeline)
        finally:
            log(f"stopping {engine}")
            compose(["--profile", engine, "down", "--remove-orphans"], env, check=False)
            wait_vram_free()

        # Copy the container's own log out before it is removed on the next up.
        logs = compose(["--profile", engine, "logs", "--no-color", engine], env,
                       check=False, capture_output=True, text=True)
        if logs.stdout:
            (out / f"{engine}.{config}.server.log").write_text(logs.stdout)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(BENCH_DIR / "spec-v1.toml"))
    ap.add_argument("--only", action="append", choices=["vllm", "nano-vllm"],
                    help="restrict to one engine (repeatable)")
    ap.add_argument("--scenario", action="append", help="restrict to one scenario (repeatable)")
    ap.add_argument("--out", help="results directory (default: experiments/<auto>)")
    ap.add_argument("--grafana", default="http://localhost:3000")
    ap.add_argument("--no-obs", action="store_true", help="skip prometheus/grafana/dcgm")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = tomllib.loads(spec_path.read_text())

    engines = args.only or ["vllm", "nano-vllm"]
    scenarios = spec["scenarios"]
    if args.scenario:
        scenarios = [s for s in scenarios if s["name"] in args.scenario]
        if not scenarios:
            raise SystemExit(f"no scenario matched {args.scenario}")

    slug = re.sub(r"[^a-z0-9]+", "-", spec["model"]["id"].lower()).strip("-")
    sha = git("rev-parse", "--short=7", "HEAD")
    dirty = "-dirty" if git("status", "--porcelain", "--", "src") else ""
    default_out = (REPO / "experiments" /
                   f"{datetime.now(timezone.utc):%Y-%m-%d}-{slug}-v{spec['version']}-{sha}{dirty}")
    out = Path(args.out).resolve() if args.out else default_out

    env = base_env(spec, out, spec["run"]["matched_max_num_seqs"])

    if args.dry_run:
        print(f"spec       {spec_path.name}  "
              f"sha256={hashlib.sha256(spec_path.read_bytes()).hexdigest()[:16]}…")
        print(f"model      {spec['model']['id']}@{spec['model']['revision'][:12]}")
        print(f"vllm       {spec['engines']['vllm']['image']}")
        print(f"nano-vllm  {sha}{dirty}")
        print(f"out        {out}")
        print("\nplan:")
        for engine in engines:
            for s in scenarios:
                cap = (spec["run"]["matched_max_num_seqs"] if s["config"] == "matched"
                       else spec["engines"][engine]["native_max_num_seqs"])
                print(f"  {engine:<10} {s['name']:<14} rate={str(s['request_rate']):<4} "
                      f"n={s['num_prompts']:<4} max_num_seqs={cap}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    log(f"results -> {out}")

    preflight(spec, env, scenarios)

    manifest = build_manifest(spec, spec_path, env)
    assert_same_toolchain(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["nano_vllm"]["dirty"]:
        log("WARNING: src/ is dirty — this run is not reproducible from a commit")

    if not args.no_obs:
        compose(["--profile", "obs", "up", "-d"], env)

    timeline = Timeline(out / "timeline.csv", args.grafana)
    start = now_ms()
    for i, engine in enumerate(engines):
        if i:
            log(f"engine swap: quiet {spec['run']['swap_gap_s']}s")
            time.sleep(spec["run"]["swap_gap_s"])
        sweep_engine(engine, spec, scenarios, out, out, timeline)
    timeline.record("all", "full-sweep", "-", start, now_ms(), 60)

    log(f"done in {(now_ms() - start) // 60000} min -> {out}")
    print(f"\nnext:  python3 bench/report.py {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
