"""Answer LLM: local GGUF (live path) or Bedrock Claude (extraction/eval only)."""
from __future__ import annotations

from config import settings

# Keep CoT short when thinking is on — quality from the checklist, not a novel.
_THINK_BRIEF = (
    " Inside <think>, stay under ~150 words: (1) premise check, (2) sections/"
    "definitions to cite, (3) enumeration checklist / grace-period gotchas."
    " Then close </think> and write the full cited answer."
)


def _is_qwen3(path: str) -> bool:
    return "Qwen3" in path or "Qwen_Qwen3" in path


def _qwen_formatter(llm):
    """Jinja chat formatter from GGUF metadata (supports enable_thinking)."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    template = (llm.metadata or {}).get("tokenizer.chat_template") or ""
    if "enable_thinking" not in template:
        return None
    eos_id = llm.token_eos()
    bos_id = llm.token_bos()
    eos = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
    bos = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
    return Jinja2ChatFormatter(
        template=template,
        eos_token=eos,
        bos_token=bos,
        stop_token_ids=[eos_id] if eos_id != -1 else None,
    )


class LocalLlm:
    """Open-weights GGUF via llama-cpp-python, all layers on one GPU.

    Default model comes from settings.model_path (MODEL_PATH in .env); pass model_path
    to override. n_ctx defaults to settings.llm_context_length — Gemma-2 trains at 8k
    and refuses n_ctx > n_ctx_train, so callers must pass 8192 for that family.
    """

    def __init__(self, model_path: str | None = None, n_ctx: int | None = None) -> None:
        self._model = None
        self._path = str(model_path or settings.model_path)
        self._n_ctx = n_ctx if n_ctx is not None else settings.llm_context_length
        self._formatter = None

    def _get(self):
        if self._model is None:
            import llama_cpp
            self._model = llama_cpp.Llama(
                self._path, n_gpu_layers=-1, verbose=False,
                n_ctx=self._n_ctx,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=settings.llm_main_gpu,
            )
            if _is_qwen3(self._path):
                self._formatter = _qwen_formatter(self._model)
        return self._model

    def _tokenize_prompt(self, prompt: str, added_special: bool) -> list[int]:
        llm = self._get()
        return llm.tokenize(
            prompt.encode("utf-8"), add_bos=not added_special, special=True,
        )

    def _complete(
        self, prompt_tokens: list[int], max_tokens: int, stop: list[str] | None = None,
    ) -> str:
        out = self._get().create_completion(
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            temperature=0.0,
            repeat_penalty=settings.llm_repeat_penalty,
            seed=settings.llm_seed,
            stop=stop or [],
        )
        return out["choices"][0]["text"]

    def generate(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        if max_tokens is None:
            max_tokens = settings.llm_max_tokens
        max_tokens = min(max_tokens, max(256, self._n_ctx - 512))

        if "gemma" in self._path.lower():
            messages = [{"role": "user", "content": f"{system}\n\n{prompt}"}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

        # Qwen3.x: control thinking via Jinja + optional think-token budget.
        if _is_qwen3(self._path):
            return self._generate_qwen(messages, max_tokens)

        out = self._get().create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.0,
            repeat_penalty=settings.llm_repeat_penalty, seed=settings.llm_seed,
        )
        return _strip_thinking(out["choices"][0]["message"]["content"].strip())

    def _generate_qwen(self, messages: list[dict], max_tokens: int) -> str:
        """Qwen3.x thinking control.

        - ``llm_enable_thinking=false``: Jinja hard-off (fast, weaker on assignment).
        - ``llm_think_budget<=0``: natural CoT until </think>, then answer in the
          same completion (the path that scored 11/12; slower).
        - ``llm_think_budget>0``: hard CoT cap then force-close (bounded latency;
          quality drops if the cap cuts mid-reasoning).
        """
        llm = self._get()
        enable = bool(settings.llm_enable_thinking)
        fmt = self._formatter
        if fmt is None:
            out = llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=0.0,
                repeat_penalty=settings.llm_repeat_penalty, seed=settings.llm_seed,
            )
            return _strip_thinking(out["choices"][0]["message"]["content"].strip())

        msgs = list(messages)
        if enable and msgs and msgs[0].get("role") == "system":
            msgs[0] = {
                **msgs[0],
                "content": msgs[0]["content"] + _THINK_BRIEF,
            }

        rendered = fmt(messages=msgs, enable_thinking=enable)
        prompt_tokens = self._tokenize_prompt(rendered.prompt, rendered.added_special)

        if not enable:
            text = self._complete(prompt_tokens, max_tokens)
            return _strip_thinking(text.strip())

        think_budget = int(settings.llm_think_budget)
        if think_budget <= 0:
            # Natural thinking within the answer token budget (no force-close).
            text = self._complete(prompt_tokens, max_tokens)
            return _strip_thinking(text.strip())

        think_text = self._complete(
            prompt_tokens, max(64, think_budget), stop=["</think>"],
        )
        cont = think_text + "</think>\n\n"
        cont_tokens = llm.tokenize(cont.encode("utf-8"), add_bos=False, special=True)
        answer = self._complete(prompt_tokens + cont_tokens, max_tokens)
        return _strip_thinking(answer.strip())


_CoT_MARKERS = ("\n**final answer", "\nfinal answer:", "\n**answer**", "\nanswer:")
_CoT_PREFIXES = ("**Final Answer**", "Final answer:", "**Answer**", "Answer:")


def _strip_thinking(text: str) -> str:
    """Drop think-blocks / freeform CoT preambles; keep the cited answer.

    Qwen3 fences its CoT, so ``</think>`` is the whole job there. The marker pass
    below is for models without a thinking template (Mistral/Gemma in the model
    ladder) that label a final answer instead. It only fires when at least a third
    of the text follows the marker, so an answer that merely *contains*
    "Answer:" near the end is never truncated to its own tail.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    lower = text.lower()
    for marker in _CoT_MARKERS:
        idx = lower.rfind(marker)
        if idx < 0 or len(text) - idx < len(text) // 3:
            continue
        cut = text[idx:].lstrip("\n")
        for prefix in _CoT_PREFIXES:
            if cut.lower().startswith(prefix.lower()):
                cut = cut[len(prefix):].lstrip(" :\n")
                break
        return cut.strip()
    if text.startswith("Here's a thinking process"):
        parts = [p for p in text.split("\n\n") if "[" in p and "]" in p]
        if parts:
            return "\n\n".join(parts[-4:]).strip()
    return text


class BedrockLlm:
    """Claude via Bedrock converse API. NOT used in the live answer path."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.bedrock_model
        self._client = None

    def _get(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        return self._client

    def generate(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        out = self._get().converse(
            modelId=self._model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return out["output"]["message"]["content"][0]["text"].strip()


def answer_llm():
    return LocalLlm() if settings.llm_backend == "local" else BedrockLlm()
