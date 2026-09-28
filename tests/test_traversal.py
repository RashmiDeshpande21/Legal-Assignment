"""As-of version resolution, restate classification, and edge-type normalisation."""
from __future__ import annotations

from datetime import date

import networkx as nx
import pytest

from graph.schema import EDGE_AMENDS, EDGE_CONTAINS, EDGE_REFERENCES
from graph.traversal import (
    _surgical_restate,
    amendment_catalog,
    amends_targets,
    definition_closure,
    edge_kind,
    find_effective,
    hop_neighbors,
    party_roster,
    sibling_neighbors,
)

FIRST = date(2018, 6, 26)
SECOND = date(2020, 5, 13)


def _mini_graph() -> nx.MultiDiGraph:
    """base::7.05 restated by a First Amendment, plus a sibling §7.06."""
    G = nx.MultiDiGraph()
    G.add_node("base", node_type="Document", name="Credit Agreement", date="2017-10-26")
    G.add_node("amendment_1", node_type="Document", name="First Amendment",
               date=FIRST.isoformat())
    G.add_node("base::art7", node_type="Section", number="VII", title="Covenants",
               text="", document_id="base", level="article")
    for num in ("7.05", "7.06"):
        G.add_node(f"base::{num}", node_type="Section", number=num, title="",
                   text=f"Original body of {num}. " * 40, document_id="base",
                   level="section")
        G.add_edge("base::art7", f"base::{num}", key=EDGE_CONTAINS,
                   edge_type=EDGE_CONTAINS)
    return G


def _amend(G, target, op_type, text, eff=FIRST, instrument="First Amendment"):
    G.add_edge("amendment_1", target, key=f"{EDGE_AMENDS}::amendment_1::1.1",
               edge_type=EDGE_AMENDS, effective_date=eff.isoformat(),
               instrument=instrument, amendment_type=op_type, amended_text=text,
               source_sec_id="amendment_1::1.1")


class TestEdgeKind:
    def test_reads_edge_type_attribute(self):
        assert edge_kind("REFERENCES", {"edge_type": EDGE_REFERENCES}) == EDGE_REFERENCES

    def test_falls_back_to_key_prefix_for_legacy_artifacts(self):
        # graph.json built before edge_type was written on every edge.
        assert edge_kind("AMENDS::amendment_2::1.5", {}) == EDGE_AMENDS
        assert edge_kind("CONTAINS", None) == EDGE_CONTAINS


class TestFindEffective:
    def test_original_text_before_effective_date(self):
        G = _mini_graph()
        _amend(G, "base::7.05", "restate", "Restated body. " * 60)
        eff = find_effective(G, "base::7.05", date(2018, 1, 1))
        assert eff.status == "original"
        assert "Restated body" not in eff.text
        assert "as originally executed (2017-10-26)" in eff.citation

    def test_amended_text_on_and_after_effective_date(self):
        G = _mini_graph()
        _amend(G, "base::7.05", "restate", "Restated body. " * 60)
        for when in (FIRST, date(2021, 1, 1)):
            eff = find_effective(G, "base::7.05", when)
            assert eff.status == "amended", when
            assert "Restated body" in eff.text
            assert "as amended by First Amendment" in eff.citation

    def test_as_of_none_resolves_to_latest(self):
        G = _mini_graph()
        _amend(G, "base::7.05", "restate", "Restated body. " * 60)
        assert "Restated body" in find_effective(G, "base::7.05", None).text

    def test_insert_appends_and_keeps_base_body(self):
        G = _mini_graph()
        _amend(G, "base::7.05", "insert", "Notwithstanding the foregoing, suspended.")
        eff = find_effective(G, "base::7.05", SECOND)
        assert "Original body of 7.05" in eff.text
        assert "suspended" in eff.text

    def test_surgical_restate_annotates_instead_of_replacing(self):
        """A short amendment instruction must not wipe clause-level carve-outs."""
        G = _mini_graph()
        _amend(G, "base::7.05", "restate",
               "by replacing the text in the proviso with the following")
        eff = find_effective(G, "base::7.05", SECOND)
        assert "Original body of 7.05" in eff.text
        assert "proviso/restate per First Amendment" in eff.text


class TestSurgicalRestate:
    def test_empty_new_text_is_surgical(self):
        assert _surgical_restate("body " * 200, "") is True

    def test_short_instruction_is_surgical(self):
        assert _surgical_restate("body " * 200, "by deleting clause (iv)") is True

    def test_full_length_substitute_is_a_real_restatement(self):
        assert _surgical_restate("body " * 100, "new body " * 100) is False

    def test_missing_base_text_means_amendment_supplies_the_body(self):
        assert _surgical_restate("", "new body " * 100) is False


class TestSiblingNeighbors:
    def test_finds_sibling_through_contains_parent(self):
        """§8.01 -> §8.02 is the remedies hop that downward-only DFS cannot make."""
        G = _mini_graph()
        assert sibling_neighbors(G, "base::7.05") == ["base::7.06"]

    def test_excludes_self(self):
        G = _mini_graph()
        assert "base::7.05" not in sibling_neighbors(G, "base::7.05")

    def test_respects_limit_and_unknown_nodes(self):
        G = _mini_graph()
        assert sibling_neighbors(G, "base::7.05", limit=0) == []
        assert sibling_neighbors(G, "base::nope") == []


class TestRealGraph:
    """Contracts the shipping pipeline depends on, asserted on the built graph."""

    def test_every_edge_declares_its_type(self, G):
        untyped = [
            (u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
            if not d.get("edge_type")
        ]
        assert untyped == [], f"{len(untyped)} edges missing edge_type"

    def test_remedies_section_is_a_sibling_of_events_of_default(self, G):
        assert "base::8.02" in sibling_neighbors(G, "base::8.01", limit=None)

    def test_event_of_default_definition_is_a_hub_to_remedies(self, G):
        """The other route to §8.02: both §8.01 and §8.02 reference the definition."""
        hub = "base::def::Event of Default"
        assert hub in hop_neighbors(G, "base::8.01", limit=None)
        assert "base::8.02" in hop_neighbors(G, hub, limit=None)

    def test_party_roster_covers_every_role(self, G):
        roster = party_roster(G)
        by_role = {r["role"] for r in roster}
        assert {"Borrower", "Parent", "Guarantor", "Administrative Agent",
                "L/C Issuer", "Co-Syndication Agent",
                "Co-Documentation Agent"} <= by_role
        names = {r["name"] for r in roster}
        assert any("DENNY" in n and "INC" in n.upper() for n in names)

    def test_amendment_catalog_is_complete_and_ordered(self, G):
        rows = amendment_catalog(G)
        assert len(rows) == 68
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["instrument"]] = counts.get(r["instrument"], 0) + 1
        assert counts == {"First Amendment": 4, "Second Amendment": 30,
                          "Third Amendment": 34}
        dates = [r["effective_date"] for r in rows]
        assert dates == sorted(dates)

    @pytest.mark.parametrize(
        ("as_of", "expect_amended"),
        [(date(2018, 1, 1), False), (date(2020, 6, 1), True)],
    )
    def test_restricted_payments_flips_at_the_amendment_boundary(
        self, G, as_of, expect_amended,
    ):
        eff = find_effective(G, "base::7.05(a)", as_of)
        assert (eff.status != "original") is expect_amended


class TestAmendmentInsertedSections:
    """Sections a later instrument adds to the agreement must become real nodes.

    §1.08 (Divisions) and §10.23 (Supported QFCs) are introduced by the Second
    Amendment as "Article X ... amended by inserting the following new Section ...",
    a phrasing the section-level pattern cannot see, so both were absent from the
    graph entirely and unanswerable at any as-of date.
    """

    @pytest.mark.parametrize(
        "node_id", ["base::1.08", "base::1.09", "base::7.19", "base::10.23"],
    )
    def test_inserted_section_exists_and_resolves_to_its_text(self, G, node_id):
        assert G.has_node(node_id)
        assert len(find_effective(G, node_id, date(2024, 1, 1)).text) > 200

    def test_inserted_section_is_absent_before_its_effective_date(self, G):
        assert find_effective(G, "base::10.23", date(2018, 1, 1)).text == ""

    def test_inserted_section_is_attached_to_the_base_agreement(self, G):
        assert G.nodes["base::10.23"]["document_id"] == "base"


class TestDefinitionClosure:
    """The DEPENDS_ON read-through a relevance ranker cannot recover."""

    def test_returns_the_terms_a_ratio_is_computed_from(self, G):
        closure = definition_closure(G, "base::def::Consolidated Leverage Ratio")
        assert {"base::def::Consolidated EBITDA",
                "base::def::Consolidated Funded Indebtedness"} <= set(closure)

    def test_excludes_the_term_itself(self, G):
        term = "base::def::Consolidated EBITDA"
        assert term not in definition_closure(G, term, depth=2)

    def test_stays_small_enough_to_force_include(self, G):
        """Force-inclusion is only safe while closures cannot flood the prompt."""
        terms = [n for n, d in G.nodes(data=True) if d["node_type"] == "Definition"]
        assert max(len(definition_closure(G, t)) for t in terms) <= 12

    def test_depth_zero_expands_nothing(self, G):
        assert definition_closure(G, "base::def::Consolidated Leverage Ratio", 0) == []

    def test_deeper_closures_are_supersets(self, G):
        term = "base::def::Consolidated Leverage Ratio"
        assert set(definition_closure(G, term, 1)) <= set(definition_closure(G, term, 2))

    def test_tolerates_unknown_and_non_definition_nodes(self, G):
        assert definition_closure(G, "base::does-not-exist") == []
        assert definition_closure(G, "base::7.01") == []


class TestAmendedSectionCoverage:
    """Every section the instruments say they amend must be an AMENDS target."""

    @pytest.mark.parametrize(
        "node_id",
        ["base::1.01",        # "Section 1.01 ... amended by inserting the following definitions"
         "base::2.15(a)(iv)",  # nested subsection: "Section 2.15(a)(iv) ... is hereby amended"
         "base::10.23",        # "Article X ... inserting the following new Section 10.23"
         "base::1.08"],
    )
    def test_section_is_recorded_as_amended(self, G, node_id):
        assert node_id in {r["target_id"] for r in amendment_catalog(G)}


class TestAmendsTargets:
    """The amendment clause -> base provision hop that carries no text of its own."""

    def test_resolves_an_inserting_clause_to_the_section_it_creates(self, G):
        assert amends_targets(G, "amendment_2::1.10") == ["base::1.08"]
        assert amends_targets(G, "amendment_2::1.22") == ["base::10.23"]

    def test_every_amends_edge_names_a_clause_that_exists(self, G):
        for _, _, k, d in G.edges(keys=True, data=True):
            if edge_kind(k, d) != EDGE_AMENDS:
                continue
            src = d.get("source_sec_id")
            assert src and G.has_node(src)
            assert amends_targets(G, src)

    def test_a_base_section_amends_nothing(self, G):
        assert amends_targets(G, "base::7.01") == []
