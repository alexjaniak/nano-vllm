#!/usr/bin/env bash
# Head-to-head request-rate sweep: vLLM, then nano-vllm, same GPU, same load,
# one server up at a time (they'd otherwise fight over VRAM).
#
# Every phase is bracketed by a quiet gap so the Grafana timeline has visible
# dead zones between runs, and every phase's start/end epoch-ms is written to
# results/<stamp>/timeline.csv along with a ready-made Grafana URL.
#
#   bash scripts/sweep.sh
#
# Knobs (env): MODEL NUM_PROMPTS RATES MAX_NUM_SEQS INPUT_LEN OUTPUT_LEN GAP
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-8B}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
RATES="${RATES:-1 2 4 8 inf}"
# Both engines get the SAME cap. nano-vllm's scheduler already does
# iteration-level batching; what it lacks is a batched forward pass. Holding
# max_num_seqs equal is what isolates that difference instead of measuring
# vLLM's default of 256 against nano-vllm's 8.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Fixed-length random prompts, not sharegpt: output length is the dominant
# source of run-to-run variance, and pinning it makes the two curves
# comparable. Re-run with DATASET=sharegpt later for the realistic shape.
DATASET="${DATASET:-random}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
# Vast's instance portal can already own :8000; override these to dodge it.
VLLM_PORT="${VLLM_PORT:-8000}"
NANO_PORT="${NANO_PORT:-8001}"
GAP="${GAP:-45}"          # quiet seconds around every measured run
SWAP_GAP="${SWAP_GAP:-90}" # longer gap when swapping engines

GRAFANA="${GRAFANA:-http://localhost:3000}"
DASH="$GRAFANA/d/nano-vllm-vs-vllm"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/$RUN_ID"
mkdir -p "$OUT"
TIMELINE="$OUT/timeline.csv"
echo "engine,phase,rate,start_ms,end_ms,duration_s,grafana_url" > "$TIMELINE"

now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }
log()    { echo "[$(date -u +%H:%M:%SZ)] $*"; }

# ---------------------------------------------------------------------------

kill_stray_vllm() {
  # Vast's vLLM template autostarts a server under supervisord. Left running
  # it would hold most of the VRAM (--gpu-memory-utilization), so nano-vllm
  # could never load — and killing it by PID just makes supervisor restart it.
  # Stop it through supervisor first, then sweep up anything else.
  if command -v supervisorctl >/dev/null 2>&1; then
    log "supervisor services:"; supervisorctl status 2>/dev/null || true
    supervisorctl stop vllm >/dev/null 2>&1 || true
  fi
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  log "VRAM after clearing strays: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

start_vllm() {
  log "starting vLLM on :$VLLM_PORT"
  nohup vllm serve "$MODEL" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    > "$OUT/vllm.server.log" 2>&1 &
  echo $! > "$OUT/server.pid"
}

start_nano() {
  log "starting nano-vllm on :$NANO_PORT"
  nohup uv run src/main.py \
    --model "$MODEL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --host 127.0.0.1 --port "$NANO_PORT" \
    > "$OUT/nano-vllm.server.log" 2>&1 &
  echo $! > "$OUT/server.pid"
}

wait_healthy() { # port
  log "waiting for :$1/health (model load can take minutes)"
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null && { log "healthy"; return 0; }
    sleep 2
  done
  echo "ERROR: :$1 never became healthy; see $OUT/*.server.log" >&2
  exit 1
}

stop_server() {
  [ -f "$OUT/server.pid" ] || return 0
  local pid; pid="$(cat "$OUT/server.pid")"
  log "stopping server pid $pid"
  # SIGTERM lets nano-vllm's lifespan hook shut the worker process down, which
  # is what actually frees the model's VRAM. Only then is the GPU free for the
  # next engine.
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$OUT/server.pid"
  sleep 5
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

record() { # engine phase rate start_ms end_ms
  local dur=$(( ($5 - $4) / 1000 ))
  # Pad the window by one gap on each side so the Grafana view includes the
  # quiet shoulders — that's what makes a phase boundary legible.
  local from=$(( $4 - GAP * 1000 )) to=$(( $5 + GAP * 1000 ))
  # var-job matches the dashboard's `job` template variable, so the link opens
  # already filtered to the engine that ran in that window.
  echo "$1,$2,$3,$4,$5,$dur,$DASH?from=$from&to=$to&var-job=$1" >> "$TIMELINE"
}

bench() { # engine port rate
  local engine=$1 port=$2 rate=$3
  log "quiet $GAP s"; sleep "$GAP"
  local t0; t0=$(now_ms)
  log "=== $engine @ rate=$rate  start_ms=$t0"
  vllm bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:$port" \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$rate" \
    --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$OUT" \
    --result-filename "${engine}_rate${rate}.json" \
    2>&1 | tee "$OUT/${engine}_rate${rate}.log"
  local t1; t1=$(now_ms)
  log "=== $engine @ rate=$rate  end_ms=$t1"
  record "$engine" measured "$rate" "$t0" "$t1"
}

sweep() { # engine port
  local engine=$1 port=$2
  # Warmup absorbs CUDA context creation, kernel autotune and the first
  # allocator growth. Recorded as its own phase so it's easy to ignore.
  local w0; w0=$(now_ms)
  log "warmup ($engine)"
  vllm bench serve --backend openai --base-url "http://127.0.0.1:$port" \
    --model "$MODEL" --dataset-name "$DATASET" \
    --random-input-len "$INPUT_LEN" --random-output-len 32 \
    --num-prompts 16 --request-rate 4 --seed 0 \
    > "$OUT/${engine}_warmup.log" 2>&1
  record "$engine" warmup - "$w0" "$(now_ms)"
  for rate in $RATES; do bench "$engine" "$port" "$rate"; done
}

# ---------------------------------------------------------------------------

trap 'stop_server' EXIT

SWEEP_START=$(now_ms)
log "run $RUN_ID | model=$MODEL max_num_seqs=$MAX_NUM_SEQS prompts=$NUM_PROMPTS"
kill_stray_vllm

start_vllm;  wait_healthy "$VLLM_PORT"; sweep vllm "$VLLM_PORT";           stop_server
log "engine swap: quiet $SWAP_GAP s"; sleep "$SWAP_GAP"
start_nano;  wait_healthy "$NANO_PORT"; sweep nano-vllm "$NANO_PORT"; stop_server

SWEEP_END=$(now_ms)
record all full-sweep - "$SWEEP_START" "$SWEEP_END"

echo
echo "================ timeline ================"
column -s, -t < "$TIMELINE"
echo
echo "Whole run in Grafana:"
echo "  $DASH?from=$((SWEEP_START - 60000))&to=$((SWEEP_END + 60000))"
echo
echo "Results: $OUT"
