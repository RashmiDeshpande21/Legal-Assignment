"""Context packing, graph-projection blocks, and CoT stripping."""
from __future__ import annotations

from config import settings
from eval.questions import QUESTIONS
from llm.client import _strip_thinking
from retrieval.pipeline import (
    _CHANGE_WORD,
    _EFFECT_WORD,
    _catalog_context,
    _clip_blocks,
    _component_effect_context,
    _context_char_budget,
    _pack_blocks,
    _pack_reserved,
    _parties_context,
)


class TestPackBlocks:
    def test_keeps_caller_order_so_priority_blocks_survive(self):
        blocks = [(f"cit{i}", "x" * 400, "html") for i in range(5)]
        packed = _pack_blocks(blocks, 100_000)
        assert [c for c, _, _ in packed] == [c for c, _, _ in blocks]

    def test_truncates_rather_than_overflowing_the_budget(self):
        blocks = [("cit", "x" * 10_000, "html")]
        packed = _pack_blocks(blocks, 2_000)
        assert sum(len(t) for _, t, _ in packed) < 2_000
        assert packed[0][1].endswith("[truncated]")

    def test_drops_trailing_blocks_once_the_budget_is_spent(self):
        """Earlier blocks stay whole, the next is clipped, the rest are dropped."""
        blocks = [("a", "x" * 1_500, "html"), ("b", "y" * 1_500, "html"),
                  ("c", "z" * 1_500, "html")]
        packed = _pack_blocks(blocks, 2_000)
        assert [c for c, _, _ in packed] == ["a", "b"]
        assert packed[0][1] == "x" * 1_500
        assert packed[1][1].endswith("[truncated]")
        assert sum(len(t) for _, t, _ in packed) < 2_000

    def test_noop_on_empty_input_or_zero_budget(self):
        assert _pack_blocks([], 1_000) == []
        assert _pack_blocks([("a", "x", "html")], 0) == [("a", "x", "html")]


class TestPackReserved:
    def test_filler_cannot_evict_a_reserved_block(self):
        reserved = [(
            ("§8.02 remedies", "terminate, accelerate, cash collateral", "html"),
            ["base::8.02"],
        )]
        fill = [(("§10.08", "x" * 5_000, "html"), ["base::10.08"])]
        packed, ids = _pack_reserved(reserved, fill, 800)
        assert packed[0][1] == "terminate, accelerate, cash collateral"
        assert ids[0] == "base::8.02"
        assert "base::10.08" not in ids or packed[-1][1].endswith("[truncated]")

    def test_every_reserved_node_survives_a_budget_that_cannot_hold_them_whole(self):
        """Shortening is allowed. Omitting a reserved node is not."""
        reserved = [
            ((f"§{i}", "word " * 400, "html"), [f"base::{i}"]) for i in range(6)
        ]
        packed, ids = _pack_reserved(reserved, [], 1_200)
        assert ids == [f"base::{i}" for i in range(6)]
        assert len(packed) == 6
        assert all(b[1] for b in packed)

    def test_a_catalog_is_not_split_across_the_per_node_share(self):
        catalog = (("Amendment catalog", "row " * 30, "html"), ["base::5.23", "base::7.05"])
        node = (("§8.02", "remedies", "html"), ["base::8.02"])
        packed, ids = _pack_reserved([catalog, node], [], 5_000)
        assert packed[0][1].startswith("row ")
        assert "base::5.23" in ids and "base::8.02" in ids


class TestClipBlocks:
    def test_clips_long_bodies_and_leaves_short_ones_alone(self):
        clipped = _clip_blocks([("a", "x" * 500, "html"), ("b", "y" * 10, "html")], 100)
        assert len(clipped[0][1]) <= 100
        assert clipped[1][1] == "y" * 10


class TestContextCharBudget:
    def test_reserves_room_for_thinking_plus_answer_tokens(self):
        """prompt + CoT + answer must fit n_ctx, or llama.cpp overflows mid-answer."""
        budget = _context_char_budget()
        reserve = settings.llm_max_tokens + max(0, settings.llm_think_budget) + 512
        assert budget <= (settings.llm_context_length - reserve) * 3
        assert budget >= 4_000


class TestGraphProjectionBlocks:
    def test_catalog_groups_by_instrument_and_states_the_total(self, G):
        blocks, _ = _catalog_context(G)
        (citation, text, _), = blocks
        assert "AMENDS" in citation
        assert "Total amendment operations: 68" in text
        for instrument, n in [("First Amendment", 4), ("Second Amendment", 30),
                              ("Third Amendment", 34)]:
            assert f"{instrument} — effective" in text
            assert f"{n} operations" in text

    def test_catalog_does_not_repeat_the_citation_label_per_row(self, G):
        """Row-level label repetition trips repeat_penalty and drops rows (§5.23)."""
        blocks, _ = _catalog_context(G)
        (_, text, _), = blocks
        assert text.count("Amendment catalog") == 0
        assert "§5.23" in text

    def test_catalog_reports_the_nodes_it_put_in_context(self, G):
        """Otherwise Tier-1 scores Q11 at 0.0 while the answer lists every row."""
        _, ids = _catalog_context(G)
        assert "base::5.23" in ids
        assert "base::def::Consolidated EBITDA" in ids
        assert len(ids) == len(set(ids))

    def test_party_roster_block_names_parties_and_returns_node_ids(self, G):
        blocks, ids = _parties_context(G)
        (_, text, _), = blocks
        assert "Borrower" in text and "Administrative Agent" in text
        assert len(ids) >= 9
        assert all(i.startswith("party::") for i in ids)


class TestComponentEffect:
    def test_only_q12_in_the_assignment_set_asks_for_it(self):
        hits = [
            q.id for q in QUESTIONS
            if _CHANGE_WORD.search(q.text) and _EFFECT_WORD.search(q.text)
        ]
        assert hits == ["q12"]

    def test_block_names_the_first_amendment_ebitda_restatement(self, G):
        blocks = _component_effect_context(
            G, ["base::def::Consolidated Leverage Ratio"], None,
        )
        assert len(blocks) == 1
        text = blocks[0][1]
        assert "Consolidated EBITDA" in text
        assert "First Amendment" in text
        assert "does not undo" in text


class TestStripThinking:
    def test_drops_a_fenced_think_block(self):
        assert _strip_thinking("<think>\nreasoning\n</think>\n\nThe answer [§7.05].") == (
            "The answer [§7.05]."
        )

    def test_cuts_a_labelled_final_answer_preamble(self):
        text = "Step 1 consider.\nStep 2 consider.\nFinal answer: New York law [§10.14]."
        assert _strip_thinking(text) == "New York law [§10.14]."

    def test_never_truncates_an_answer_that_merely_mentions_answer(self):
        """A trailing "Answer:" inside real content must not eat the whole answer."""
        body = "New York law governs [§10.14(a)]. " * 20
        text = body + "\nAnswer: yes."
        assert _strip_thinking(text).startswith("New York law governs")

    def test_passes_through_a_clean_answer(self):
        assert _strip_thinking("Plain cited answer [§1.01].") == (
            "Plain cited answer [§1.01]."
        )
