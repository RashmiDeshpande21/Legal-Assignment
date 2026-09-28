"""Seeding and the two-stage expansion, graph-only (no embedder/LLM)."""
from __future__ import annotations

import re

import pytest

from config import settings
from eval.questions import QUESTIONS
from graph.intent import find_seeds
from retrieval.pipeline import _expand_seeds_with_hops, _stage2_candidates


class TestFindSeeds:
    def test_picks_up_explicit_section_references(self, G):
        seeds = find_seeds(G, "What does Section 7.05(a) permit?")
        assert "base::7.05(a)" in seeds

    @pytest.mark.parametrize(
        "cite", ["Section 7.10", "section 7.10", "§7.10", "§ 7.10", "Sec. 7.10",
                 "Sections 7.10", "§§7.10"],
    )
    def test_recognises_the_ways_a_section_is_actually_cited(self, G, cite):
        """A reviewer writing "§7.10" must not silently get zero section seeds."""
        assert "base::7.10" in find_seeds(G, f"What does {cite} require?")

    def test_falls_back_to_the_nearest_modelled_ancestor(self, G):
        """Questions can cite deeper than the graph models the provision."""
        assert "base::7.10(a)" in find_seeds(G, "Does §7.10(a)(iii) still apply?")

    def test_ignores_section_numbers_that_do_not_exist(self, G):
        assert find_seeds(G, "What does Section 99.99 say?") == []

    def test_matches_defined_terms_in_the_question(self, G):
        seeds = find_seeds(G, "How is Consolidated EBITDA calculated?")
        assert "base::def::Consolidated EBITDA" in seeds

    def test_skips_ultra_generic_defined_terms(self, G):
        """"the Borrower"/"the agreement" match almost every question and hop-explode."""
        seeds = find_seeds(G, "May the Borrower under this agreement grant a lien?")
        assert "base::def::Borrower" not in seeds
        assert "base::def::Agreement" not in seeds

    def test_returns_unique_seeds_in_priority_order(self, G):
        seeds = find_seeds(G, "Section 7.05 and Section 7.05 again")
        assert seeds == list(dict.fromkeys(seeds))

    def test_no_seeds_for_an_off_corpus_question(self, G):
        assert find_seeds(G, "What is the weather in Madrid?") == []

    def test_matches_defined_terms_cited_colloquially(self, G):
        """Unseen questions say "the leverage ratio", not "Consolidated Leverage Ratio"."""
        seeds = find_seeds(G, "Has the definition of the leverage ratio changed?")
        assert "base::def::Consolidated Leverage Ratio" in seeds

    def test_a_bare_qualifier_does_not_seed_every_term_it_prefixes(self, G):
        """Sub-phrase matching must stay a phrase match, not keyword search."""
        seeds = find_seeds(G, "Are the consolidated figures audited?")
        assert not [s for s in seeds if s.startswith("base::def::Consolidated")]

    def test_seeding_stays_selective_across_the_question_set(self, G):
        """A seed explosion silently turns the graph path into whole-corpus retrieval."""
        for q in QUESTIONS:
            assert len(find_seeds(G, q.retrieval_text or q.text)) <= 4, q.id


class TestStageOneExpansion:
    def test_keeps_seeds_first_and_adds_hop_neighbours(self, G):
        expanded, force = _expand_seeds_with_hops(G, ["base::def::Borrowing"])
        assert expanded[0] == "base::def::Borrowing"
        assert len(expanded) > 1
        assert force[0] == "base::def::Borrowing"

    def test_tolerates_unknown_nodes(self, G):
        expanded, force = _expand_seeds_with_hops(G, ["base::does-not-exist"])
        assert expanded == ["base::does-not-exist"]
        assert force == ["base::does-not-exist"]


class TestStageTwoCandidates:
    def test_reaches_the_remedies_section_from_events_of_default(self, G):
        """The q7 gap: §8.02 is a sibling of §8.01, invisible to downward DFS."""
        assert "base::8.02" in _stage2_candidates(G, ["base::8.01"], top_n=1)

    def test_reaches_ratio_components_through_the_definition_hub(self, G):
        """The q12 gap: the leverage ratio's inputs are one DEPENDS_ON hop away."""
        cands = _stage2_candidates(G, ["base::def::Consolidated Leverage Ratio"],
                                   top_n=1)
        assert "base::def::Consolidated EBITDA" in cands

    def test_does_not_truncate_neighbours_alphabetically(self, G):
        """Capping per node dropped "Permitted Liens" off the end of Q5's list."""
        assert "base::def::Permitted Liens" in _stage2_candidates(
            G, ["base::def::Lien"], top_n=1,
        )

    def test_never_re_adds_evidence_it_was_given(self, G):
        evidence = ["base::8.01", "base::8.02"]
        assert not set(_stage2_candidates(G, evidence, top_n=2)) & set(evidence)

    def test_expands_only_the_first_top_n_evidence_nodes(self, G):
        assert _stage2_candidates(G, ["base::8.01"], top_n=0) == []
        assert _stage2_candidates(G, [], top_n=settings.expand_stage2_top_n) == []

    def test_all_candidates_exist_in_the_graph(self, G):
        cands = _stage2_candidates(G, ["base::7.10", "base::8.01"], top_n=2)
        assert cands and all(G.has_node(n) for n in cands)

    def test_expands_from_an_explicit_subset_when_given_one(self, G):
        """Seeds are expanded even when they rank below top_n in the combined list."""
        evidence = ["base::7.05(a)", "base::7.03", "base::7.04(e)",
                    "base::6.01", "base::5.02", "base::5.03", "base::8.01"]
        assert "base::8.02" not in _stage2_candidates(G, evidence, top_n=6)
        assert "base::8.02" in _stage2_candidates(
            G, evidence, top_n=6, expand_from=["base::8.01"],
        )


class TestQuestionSet:
    def test_twelve_verbatim_questions(self):
        assert len(QUESTIONS) == 12
        assert [q.id for q in QUESTIONS] == [f"q{i}" for i in range(1, 13)]

    def test_only_q9_and_q10_are_date_dependent(self):
        dated = {q.id for q in QUESTIONS if q.as_of}
        assert dated == {"q9", "q10"}

    def test_q10_retrieval_text_resolves_the_anaphora(self):
        q10 = next(q for q in QUESTIONS if q.id == "q10")
        assert q10.retrieval_text and "restricted payment" in q10.retrieval_text.lower()
        assert re.search(r"2020-06-01", q10.retrieval_text)
