"""Shared fixtures. The built graph is loaded once per session (read-only in tests)."""
from __future__ import annotations

from pathlib import Path

import pytest

from config import settings
from graph.loader import load_graph

GRAPH_PATH = settings.artifacts_dir / "graph.json"


@pytest.fixture(scope="session")
def G():
    if not GRAPH_PATH.exists():
        pytest.skip(f"{GRAPH_PATH} missing — run `make build` first")
    return load_graph(GRAPH_PATH)


@pytest.fixture(scope="session")
def eval_set() -> list[dict]:
    import json
    return json.loads(Path("experiments/mini_eval_set.json").read_text())
