"""Rule-based extraction: amendment ops, parties, cross-references, exceptions, term deps.

All patterns verified against the actual documents; no LLM in this path.
"""
from __future__ import annotations

import logging
import re

from graph.schema import ORDINALS, AmendOp, Definition, Document, Party, Section

logger = logging.getLogger(__name__)

FOOTER_RE = re.compile(r"CHAR1\\\d+v\d+")
DEF_TARGET_RE = re.compile(r"definition of \u201c([^\u201d]+)\u201d")
SEC_TARGET_RE = re.compile(r"Section (\d{1,2}\.\d{2}(?:\([a-z0-9]+\))?)")
EXHIBIT_TARGET_RE = re.compile(r"Exhibit ([A-Z])\b")
SEC_REF_RE = re.compile(r"Section[s]? (\d{1,2}\.\d{2}(?:\([a-z0-9]+\))?)")
PARTY_LINE_RE = re.compile(r"^(.+?),? as (?:the )?(.+?),?$")


def classify_op(text: str) -> str:
    # Proviso / clause surgeries quote "replacing" without being full restatements.
    if re.search(
        r"(?:amended by (?:deleting|replacing)|by replacing the text in the proviso|"
        r"amended by replacing)",
        text, re.I | re.S,
    ):
        return "replace"
    if re.search(r"amended by (?:inserting|adding)", text):
        return "insert"
    if re.search(r"amended by deleting", text):
        return "delete"
    return "restate"  # "amended and restated in its entirety" and default


_DEF_AMEND_RE = re.compile(
    r"(?:The d|D)efinition of [\u201c\"]([^\u201d\"]+)[\u201d\"]"
    r".{0,120}?is hereby (amended|added|deleted)",
    re.I | re.S,
)
# First Amendment style: "Clause (b)(iv) of the definition of “X” in Section … is hereby"
_CLAUSE_DEF_AMEND_RE = re.compile(
    r"(?:Clause|clause)\s+\([^)]+\)\s+of the definition of [\u201c\"]([^\u201d\"]+)[\u201d\"]"
    r".{0,120}?is hereby (amended|added|deleted)",
    re.I | re.S,
)
# Nested subsections repeat the paren group: "Section 2.15(a)(iv) of the Credit
# Agreement is hereby amended". Matching a single group left "(iv) of the Credit..."
# unconsumed, so the whole clause was skipped and §2.15(a) looked untouched.
_SEC_AMEND_RE = re.compile(
    r"Section ([\d.]+(?:\([a-z0-9]+\))*) of the Credit Agreement is hereby "
    r"(amended|added|deleted|restated)",
    re.I,
)
_EXH_AMEND_RE = re.compile(
    r"Exhibit ([A-Z])\b.{0,80}?is hereby (amended|restated)",
    re.I | re.S,
)
# Wholly new sections are hung off the *article*: "Article X of the Credit Agreement is
# hereby amended by inserting the following new Section 10.23 after Section 10.22".
# _SEC_AMEND_RE cannot see these (it wants "Section X ... is hereby amended"), which is
# how §1.08 Divisions and §10.23 Supported QFCs were missing from the graph entirely.
_NEW_SEC_INSERT_RE = re.compile(
    r"inserting the following new Section\s+(\d{1,2}\.\d{1,2}(?:\([a-z0-9]+\))?)",
    re.I,
)
_INSERT_DEFS_RE = re.compile(
    r"Section 1\.01 of the Credit Agreement is hereby amended by inserting "
    r"the following definitions?",
    re.I,
)
_QUOTED_TERM_MEANS_RE = re.compile(r"[\u201c\"]([^\u201d\"]+)[\u201d\"]\s+means\b")


def extract_amend_ops(sections: list[Section], doc: Document) -> list[AmendOp]:
    """Pull AmendOps from amendment Article I (and OCR blobs that embed later 1.x ops).

    Finds every 'hereby amended' clause in a section (definition / section / exhibit /
    batch definition inserts). Trailing OCR commas on term names are stripped.
    """
    instrument = f"{ORDINALS[(doc.amendment_number or 1) - 1]} Amendment"
    ops: list[AmendOp] = []
    seen: set[tuple[str, str]] = set()  # (kind, target)

    def _add(target: str, kind: str, op_type: str, new_text: str, source_sec_id: str) -> None:
        target = target.strip().rstrip(",;.")
        if not target:
            return
        key = (kind, target)
        if key in seen:
            return
        seen.add(key)
        ops.append(AmendOp(
            target=target, target_kind=kind, op_type=op_type,  # type: ignore[arg-type]
            new_text=new_text, eff_date=doc.date, instrument=instrument,
            source_sec_id=source_sec_id,
        ))

    def _new_text_after(text: str, end: int) -> str:
        after = text[end:]
        cut = re.search(r"(?:follows|thereof|the following|order)\s*:", after)
        return after[cut.end():].strip() if cut else after[:2000].strip()

    for sec in sections:
        is_article_i = sec.number.startswith("1.")
        is_amend_block = bool(re.match(r"\d+\.\d+\s+Amendment to\b", sec.text.strip()))
        if sec.level != "section" or not (is_article_i or is_amend_block):
            continue
        text = FOOTER_RE.sub("", sec.text)
        if "hereby amended" not in text and "hereby add" not in text:
            continue

        # Batch §1.01 definition inserts (Benchmark Replacement family, etc.)
        if _INSERT_DEFS_RE.search(text):
            for term in _QUOTED_TERM_MEANS_RE.findall(text):
                _add(term, "definition", "insert", text, sec.id)
            # The clause says "Section 1.01 ... is hereby amended", so §1.01 is itself
            # an amended section: without this, "which sections were amended" omits the
            # definitions section every instrument touches.
            _add("1.01", "section", "insert", text, sec.id)

        for cre in (_CLAUSE_DEF_AMEND_RE, _DEF_AMEND_RE):
            for m in cre.finditer(text):
                _add(
                    m.group(1), "definition",
                    classify_op(text[m.start(): m.start() + 240]),
                    _new_text_after(text, m.end()), sec.id,
                )

        for m in _SEC_AMEND_RE.finditer(text):
            window = text[m.start(): m.start() + 180]
            if _INSERT_DEFS_RE.search(window):
                continue
            pre = text[max(0, m.start() - 80): m.start()]
            if re.search(r"definition of [\u201c\"][^\u201d\"]+[\u201d\"]\s+in\s+$", pre, re.I):
                continue
            _add(
                m.group(1), "section",
                classify_op(text[m.start(): m.start() + 240]),
                _new_text_after(text, m.end()), sec.id,
            )

        for m in _NEW_SEC_INSERT_RE.finditer(text):
            _add(m.group(1), "section", "insert", _new_text_after(text, m.end()), sec.id)

        for m in _EXH_AMEND_RE.finditer(text):
            _add(
                m.group(1), "exhibit",
                classify_op(text[m.start(): m.start() + 240]),
                _new_text_after(text, m.end()), sec.id,
            )

    return ops


def extract_parties(preamble_text: str) -> list[tuple[Party, list[str]]]:
    """(party, roles) from the cover-page 'NAME, as Role' lines; aliases from the body."""
    out: list[tuple[Party, list[str]]] = []
    for line in preamble_text.split("\n"):
        line = line.strip().rstrip(",")
        m = PARTY_LINE_RE.match(line)
        if not m or len(m.group(1)) > 60 or len(m.group(2)) > 60 or not m.group(2)[0].isupper():
            continue
        names = re.split(r" and (?=[A-Z]{2})", m.group(1))
        roles = [r.strip().rstrip("s") if r.strip().endswith("s") else r.strip()
                 for r in m.group(2).split(" and ")]
        for name in names:
            out.append((Party(name=name.strip()), roles))
    if "Other Lenders" in preamble_text:
        out.append((Party(name="The Other Lenders Party Hereto"), ["Lender"]))
    for party, _ in out:
        short = party.name.split(",")[0]
        quoted = [
            q.strip()
            for w in re.finditer(re.escape(short) + r"[^\n]{0,240}", preamble_text)
            for q in re.findall(r"\u201c\s*([^\u201d]{2,40})\s*\u201d", w.group(0))
        ]
        aliases = {short, *quoted}
        aliases |= {a.replace("\u2019", "").replace("'", "") for a in aliases}
        party.aliases = sorted(aliases - {party.name})
    all_roles = {r for _, roles in out for r in roles}
    for party, roles in out:
        party.aliases = [a for a in party.aliases if a not in all_roles or a in roles]
    return out


def extract_references(sections: list[Section]) -> list[tuple[str, str, str]]:
    """(source_section_id, target_section_number, evidence) for explicit Section X.XX refs."""
    known = {s.number for s in sections}
    refs = []
    for sec in sections:
        for m in SEC_REF_RE.finditer(sec.text):
            num = m.group(1)
            target = num if num in known else num.split("(")[0]
            if target in known and target != sec.number:
                evidence = sec.text[max(0, m.start() - 60):m.end() + 20].strip()
                refs.append((sec.id, target, evidence))
    return refs


EXCEPTION_INTRO_RE = re.compile(r"(other than the following|except\b|other than\b)", re.I)


def extract_exceptions(sections: list[Section]) -> list[tuple[str, str, str]]:
    """(subsection_id, parent_section_id, evidence): lettered carve-outs under prohibitions."""
    by_id = {s.id: s for s in sections}
    out = []
    for sec in sections:
        if sec.level != "subsection":
            continue
        parent_id = sec.id.split("(")[0]
        parent = by_id.get(parent_id)
        if parent and EXCEPTION_INTRO_RE.search(parent.text) and parent.text.rstrip().endswith(":"):
            out.append((sec.id, parent_id, parent.text[-120:].strip()))
    return out


def extract_term_deps(defs: list[Definition]) -> list[tuple[str, str]]:
    """(term, depends_on_term): defined terms mentioned inside other definitions.

    Longest-first masking so 'Consolidated Net Income' is not double-counted as shorter terms.
    """
    terms = sorted((d.term for d in defs), key=len, reverse=True)
    alternation = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in terms))
    deps = []
    for d in defs:
        body = d.text[len(d.term) + 2:]  # skip the term's own heading
        seen = {m.group(0) for m in alternation.finditer(body)}
        deps.extend((d.term, t) for t in seen if t != d.term)
    return deps


def extract_section_term_refs(
    sections: list[Section], defs: list[Definition]
) -> list[tuple[str, str]]:
    """(section_id, term): defined terms used in operative sections (not §1.01 itself)."""
    terms = sorted((d.term for d in defs), key=len, reverse=True)
    alternation = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in terms))
    out = []
    for sec in sections:
        if sec.number.startswith("1.01") or sec.level == "article":
            continue
        seen = {m.group(0) for m in alternation.finditer(sec.text)}
        out.extend((sec.id, t) for t in seen)
    return out
