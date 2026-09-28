"""Extract §1.01 defined terms from the base agreement HTML.

Each definition paragraph is a leaf div starting with a cp1252 curly quote followed by
an underlined term: “<u>Term</u>” means ... Continuation paragraphs (lettered clauses,
tables) attach to the current term until the next quoted-term paragraph.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from graph.schema import Definition

TERM_OPEN_RE = re.compile(r"^\u201c(.+?)\u201d")


def extract_definitions(html: str, doc_id: str, section_id: str) -> list[Definition]:
    soup = BeautifulSoup(html, "lxml")
    defs: list[Definition] = []
    in_defs = False
    for div in soup.find_all("div"):
        if div.find("div") or div.find_parent("table"):
            continue
        txt = div.get_text(" ", strip=True).replace("\xa0", " ")
        txt = re.sub(r" {2,}", " ", txt)
        if not txt:
            continue
        if txt.startswith("1.01"):
            in_defs = True
            continue
        if in_defs and txt.startswith("1.02"):
            break
        if not in_defs:
            continue
        underlined = div.find("font", style=lambda s: s and "underline" in s)
        m = TERM_OPEN_RE.match(txt)
        if m and underlined and underlined.get_text(strip=True) == m.group(1).strip():
            defs.append(Definition(
                term=m.group(1).strip().rstrip("\u201d"),
                text=txt, document_id=doc_id, section_id=section_id,
            ))
        elif defs:
            defs[-1].text += "\n" + txt
    return defs
