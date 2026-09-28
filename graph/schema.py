"""Node dataclasses and edge type constants for the legal knowledge graph."""
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

EDGE_CONTAINS = "CONTAINS"
EDGE_DEFINES = "DEFINES"
EDGE_AMENDS = "AMENDS"
EDGE_REFERENCES = "REFERENCES"
EDGE_PARTY_TO = "PARTY_TO"
EDGE_EXCEPTION_TO = "EXCEPTION_TO"
EDGE_DEPENDS_ON = "DEPENDS_ON"

DOC_BASE = "base"
DOC_AMEND_1 = "amendment_1"
DOC_AMEND_2 = "amendment_2"

ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth",
            "Ninth", "Tenth"]


@dataclass
class Document:
    id: str
    name: str
    date: date
    type: Literal["base", "amendment"]
    amendment_number: int | None = None
    filing_url: str = ""


@dataclass
class Section:
    id: str                     # e.g. "base::7.01", "base::7.01(a)"
    number: str                 # e.g. "7.01", "7.01(a)", "VII"
    title: str
    text: str
    document_id: str
    level: Literal["article", "section", "subsection"]
    source: Literal["html", "ocr"] = "html"


@dataclass
class Definition:
    term: str
    text: str
    document_id: str
    section_id: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class Party:
    name: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class AmendOp:
    target: str                 # section number "7.04(e)", definition term, or exhibit letter
    target_kind: Literal["section", "definition", "exhibit"]
    op_type: Literal["restate", "insert", "delete", "replace"]
    new_text: str
    eff_date: date
    instrument: str             # "First Amendment" | "Second Amendment"
    source_sec_id: str          # amendment section that performs the op, e.g. "amendment_2::1.5"
