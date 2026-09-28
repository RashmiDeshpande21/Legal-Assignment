"""Every result file must record the config that produced it."""
from __future__ import annotations

import json

from config import settings
from eval.run_all import run_provenance


class TestRunProvenance:
    def test_records_the_settings_that_change_answer_content(self):
        run = run_provenance("unit")
        for key in ("model_path", "llm_enable_thinking", "llm_think_budget",
                    "llm_repeat_penalty", "llm_max_tokens", "embedder_model",
                    "reranker_model", "expand_stage2_top_n", "git_sha", "timestamp"):
            assert key in run, key
        assert run["tag"] == "unit"

    def test_is_json_serialisable(self):
        json.dumps(run_provenance("unit"))

    def test_tracks_live_settings(self):
        run = run_provenance()
        assert run["llm_repeat_penalty"] == settings.llm_repeat_penalty
        assert run["llm_think_budget"] == settings.llm_think_budget


class TestFrozenSubmission:
    def test_snapshot_is_marked_and_complete(self):
        import json
        from pathlib import Path
        frozen = json.loads(
            Path("eval/frozen/assignment_12q_answers.json").read_text()
        )
        assert frozen["run"]["frozen"] is True
        assert [r["id"] for r in frozen["graph"]] == [f"q{i}" for i in range(1, 13)]
        assert [r["id"] for r in frozen["baseline"]] == [f"q{i}" for i in range(1, 13)]
        scored = json.loads(
            Path("eval/frozen/assignment_12q_scores.json").read_text()
        )
        assert scored["graph"]["tier2"]["answer_correct_rate"] == 1.0
        assert scored["baseline"]["tier2"]["answer_correct_rate"] == 0.417


class TestSamplingDefaults:
    def test_repeat_penalty_is_off_for_long_enumerations(self):
        """1.15 over llama.cpp's 64-token window punishes repeated citation labels."""
        assert settings.llm_repeat_penalty == 1.0

    def test_answer_budget_fits_inside_the_context_window(self):
        reserve = settings.llm_max_tokens + max(0, settings.llm_think_budget)
        assert reserve < settings.llm_context_length
