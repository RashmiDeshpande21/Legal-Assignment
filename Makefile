.PHONY: setup install-model install-model-fast run build answers answers-graph eval scorecard \
        eval-all eval-clustered generate-clustered verify-citations verify-temporal \
        judge-agreement holdout test lint probe probe-holdout probe-all \
        ablation ablation-holdout

setup:                          ## Bootstrap on A10G (CUDA; nvcc must be on PATH)
	PATH="/usr/local/cuda/bin:$$PATH" CMAKE_ARGS="-DGGML_CUDA=on" \
	    uv sync --extra retrieval --extra dev --extra local-llm

install-model:                  ## Download the default generator: Qwen3.6-27B Q4_K_M (~17 GB)
	mkdir -p models
	uv run hf download bartowski/Qwen_Qwen3.6-27B-GGUF \
	    --include "Qwen_Qwen3.6-27B-Q4_K_M.gguf" --local-dir models/

install-model-fast:             ## Lower-latency alternative: Mistral-Small-24B (~14 GB, 5/12)
	mkdir -p models
	uv run hf download bartowski/Mistral-Small-24B-Instruct-2501-GGUF \
	    --include "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf" --local-dir models/
	@echo "Set MODEL_PATH=models/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf in .env"

run:                            ## Smoke test
	uv run --no-sync python answer.py --question "What is the governing law?" --as-of 2020-06-01

build:                          ## Rebuild artifacts from scratch (idempotent)
	uv run --no-sync python build_artifacts.py

TAG ?= dev

test:                           ## Unit tests (no GPU, no network; needs artifacts/)
	uv run --no-sync python -m pytest tests -q

lint:                           ## Ruff over the shipping packages
	uv run --no-sync ruff check .

probe:                          ## Tier-1 gold recall, no generator (~20s feedback loop)
	uv run --extra retrieval --no-sync python -m eval.retrieval_probe

probe-holdout:                  ## Same probe on the 20 unseen questions (generalization)
	SET=holdout uv run --extra retrieval --no-sync python -m eval.retrieval_probe

probe-all: probe probe-holdout  ## Both sets: a change must hold on the unseen one too

answers:                        ## All 12 questions, both paths -> outputs/ (TAG=<label>)
	RUN_TAG=$(TAG) uv run --extra local-llm --extra retrieval python -m eval.run_all

answers-graph:                  ## Graph path only — half the wall time while iterating
	PATHS=graph RUN_TAG=$(TAG) \
	    uv run --extra local-llm --extra retrieval python -m eval.run_all

eval:                           ## Re-judge outputs/answers.json (Bedrock). Does not touch eval/frozen/
	uv run --no-sync python -m eval.harness

scorecard:                      ## Print the frozen submission scorecard (no GPU, no judge)
	@echo "Submission snapshot: eval/frozen/ (make answers does not overwrite it)"
	@head -n 18 eval/frozen/assignment_12q_scorecard.md

eval-all:                       ## answers -> judge -> deterministic checks, one TAG
	$(MAKE) answers TAG=$(TAG)
	$(MAKE) eval
	$(MAKE) verify-citations
	$(MAKE) verify-temporal

holdout:                        ## Anti-overfitting guard: 20 unseen questions (GPU)
	HOLDOUT_OUT=experiments/results/holdout_qwen36-27b_$(TAG).json \
	    uv run --extra local-llm --extra retrieval \
	    python -m experiments.holdout_eval qwen36-27b $(TAG)

eval-clustered:                 ## Tier-1 graph vs baseline on cluster-sampled set
	uv run --extra retrieval --no-sync python -m eval.eval_clustered

generate-clustered:             ## Regenerate cluster-sampled eval set (Bedrock Opus, offline)
	uv run --no-sync python -m eval.generate_clustered_qa

verify-citations:               ## Tier 4: resolve every citation against the graph (no LLM)
	uv run --no-sync python -m eval.citation_verify

verify-temporal:                ## Tier 4: as-of resolution at every amendment boundary (no LLM)
	uv run --no-sync python -m experiments.temporal_boundary

judge-agreement:                ## Tier 4: second judge + gold-blind judge (Bedrock, ~4 min)
	uv run --no-sync python -m experiments.judge_agreement

ablation:                       ## Tier 4: per-block contribution (GPU, ~15 min per condition)
	CUDA_VISIBLE_DEVICES=1 LLM_MAIN_GPU=0 EMBED_DEVICE=cpu \
	    uv run --no-sync python -m experiments.ablation qwen36-27b

ablation-holdout:               ## Tier 4: same ablation on unsaturated holdout (GPU, ~25 min/cond)
	CUDA_VISIBLE_DEVICES=1 LLM_MAIN_GPU=0 EMBED_DEVICE=cpu \
	    uv run --no-sync python -m experiments.ablation_holdout qwen36-27b
