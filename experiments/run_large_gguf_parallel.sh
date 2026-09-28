#!/usr/bin/env bash
# One model per free A10G (skip GPU 0 — foreign vLLM). Full 12-Q eval in parallel.
# Embedder/reranker on CPU so each ~17–18GB GGUF has the full 23GB card.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/results /tmp/large_gguf_logs

# Load AWS/HF from .env without inheriting CUDA_VISIBLE_DEVICES=1,2
set -a
# shellcheck disable=SC1091
source .env
set +a

run_one() {
  local gpu="$1" model="$2"
  local out="experiments/results/large_gguf_${model}_full.json"
  local log="/tmp/large_gguf_logs/${model}_gpu${gpu}.log"
  echo "LAUNCH gpu=${gpu} model=${model} -> ${out}"
  env -u CUDA_VISIBLE_DEVICES \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    LLM_MAIN_GPU=0 \
    EMBED_DEVICE=cpu \
    LARGE_GGUF_OUT="${out}" \
    uv run python -m experiments.large_gguf_comparison "${model}" full \
    >"${log}" 2>&1 &
  echo $! >"/tmp/large_gguf_logs/${model}.pid"
}

# GPU 1 = Mistral-24B, GPU 2 = Gemma-2-27B, GPU 3 = Qwen3.6-27B
run_one 1 mistral
run_one 2 gemma2
run_one 3 qwen36

echo "PIDs: $(cat /tmp/large_gguf_logs/*.pid | tr '\n' ' ')"
echo "Logs: /tmp/large_gguf_logs/"
echo "Waiting for all three..."
wait
echo "ALL DONE"
# Merge shards into the shared results file
uv run python - <<'PY'
import json
from pathlib import Path
merged = {}
base = Path("experiments/results/large_gguf_results.json")
if base.exists():
    merged.update(json.loads(base.read_text()))
for p in sorted(Path("experiments/results").glob("large_gguf_*_full.json")):
    merged.update(json.loads(p.read_text()))
    print("merged", p.name)
base.write_text(json.dumps(merged, indent=1))
print("wrote", base)
for k, v in merged.items():
    if "correct" in v:
        print(f"  {k}: {v['correct']}/{v['of']} lat_mean="
              f"{sum(q['latency_s'] for q in v['per_question'].values())/len(v['per_question']):.1f}s "
              f"vram={v.get('vram_used_mb')} gpu={v.get('gpu')}")
    elif not v.get("ok", True):
        print(f"  {k}: FAIL {v.get('error')}")
PY
