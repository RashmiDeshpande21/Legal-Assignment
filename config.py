from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()  # exports CUDA_VISIBLE_DEVICES etc. before torch/llama_cpp init


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data/Dennys")
    artifacts_dir: Path = Path("artifacts")
    outputs_dir: Path = Path("outputs")

    aws_region: str = "us-west-2"
    bedrock_model: str = "us.anthropic.claude-opus-4-6-v1"
    bedrock_model_judge: str = "us.anthropic.claude-sonnet-4-6"

    llm_backend: str = "local"
    # Qwen3.6-27B GGUF (18GB, fits one A10G): 12/12 on the assignment set, 16/20 on the
    # held-out set, vs Qwen2.5-14B 5-7/12, Mistral-24B 5-6/12, Gemma-2-27B 6/12 on
    # identical context. Ladder detail in experiments/EXPERIMENT_SUMMARY.md §5 —
    # reasoning strength, not legal pretraining, is the bottleneck (Saul-7B, the only
    # legal-pretrained candidate, scored worst) and it only lands at >=14B.
    model_path: Path = Path("models/Qwen_Qwen3.6-27B-Q4_K_M.gguf")
    llm_main_gpu: int = 1       # visible-device index for generation (single GPU, no split)
    llm_context_length: int = 16384
    # Answer-phase new tokens after </think>. Must leave room for packed prompt
    # inside n_ctx (prompt + think_budget + max_tokens <= llm_context_length).
    llm_max_tokens: int = 4096
    # Qwen3 CoT (EXPERIMENT_SUMMARY.md §7). think_budget<=0 = natural thinking:
    # 12/12, ~25 min for 12 questions. think_budget>0 = hard CoT cap, 7/12 — truncates
    # reasoning without buying much time. enable_thinking=False = Jinja hard-off,
    # 7/12 at ~4x the speed. The soft /no_think prompt tag is ignored by this GGUF.
    llm_enable_thinking: bool = True
    llm_think_budget: int = 0
    # repeat_penalty is applied over llama.cpp's last_n_tokens_size (64) window, so
    # it punishes the repeated citation labels a long enumeration legitimately needs
    # (the 68-row AMENDS catalog silently lost §5.23 at 1.15). Greedy decoding on a 27B
    # instruct model does not need the loop guard: 1.0 = off. Seed only bites if temp > 0.
    llm_repeat_penalty: float = 1.0
    llm_seed: int = 0
    embed_device: str = "cuda:0"  # embedder + reranker live on the other GPU

    # Two-stage graph expansion (retrieval/pipeline.py). Stage 1 expands the question
    # seeds; stage 2 expands around the evidence already in the prompt — sibling
    # sections via parent CONTAINS (§8.01 -> §8.02 remedies) and bidirectional
    # REFERENCES/DEPENDS_ON hops through definition hubs. Candidates are collected
    # unbounded then cut by the reranker: truncating the neighbour list directly would
    # drop nodes alphabetically (that is how "Permitted Liens" went missing on Q5).
    expand_stage2_top_n: int = 6     # evidence nodes to expand around
    expand_stage2_keep: int = 8      # candidates kept after reranking
    definition_closure_depth: int = 1  # DEPENDS_ON read-through for a defined-term seed

    # experiment winners — see experiments/models/results/
    embedder_model: str = "Hanno-Labs/dinghy-law-0.6b-v1"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_top_k: int = 10
    reranker_batch_size: int = 32


settings = Settings()
