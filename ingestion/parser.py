"""HTML -> structured Section lists for all three Denny's documents.

Base + First Amendment are machine-readable Workiva HTML (windows-1252).
Second Amendment is scan-based; OCR text lives in hidden white 1pt font blocks.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

from graph.schema import DOC_BASE, ORDINALS, Document, Section

logger = logging.getLogger(__name__)
WHITE_FONT_RE = re.compile(r"<font[^>]*color:white[^>]*>(.*?)</font>", re.S)
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


def _doc_date(text: str) -> date:
    m = re.search(r"(?:entered into as of|dated as of)\s+(\w+ \d{1,2}, \d{4})", text)
    from datetime import datetime
    return datetime.strptime(m.group(1), "%B %d, %Y").date()


def discover_docs(data_dir: Path) -> dict[str, tuple[Document, Path, bool]]:
    """Scan data_dir for filings; derive id/type/date/OCR-ness from content, not filenames.

    Returns {doc_id: (Document, path, is_ocr)} ordered base first, then amendments by number.
    A new amendment dropped into the folder is picked up with no code change.
    """
    found = []
    for path in sorted(data_dir.glob("*.htm*")):
        html = path.read_text(encoding="windows-1252", errors="replace")
        ocr_chars = sum(len(b) for b in WHITE_FONT_RE.findall(html))
        is_ocr = ocr_chars > 5000
        text = WHITE_FONT_RE.sub(" ", html) if not is_ocr else " ".join(WHITE_FONT_RE.findall(html))
        from html import unescape
        text = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text)))
        head = text[:4000]
        am = re.search(r"\b(" + "|".join(o.upper() for o in ORDINALS) + r")\s+AMENDMENT\s+TO",
                       head)
        url_m = re.search(r"saved from url=\(\d+\)(\S+?)(?:\?\S*)?\s", html)
        url = url_m.group(1) if url_m else ""
        if am:
            n = ORDINALS.index(am.group(1).title()) + 1
            doc = Document(
                id=f"amendment_{n}",
                name=f"{ORDINALS[n - 1]} Amendment to Third Amended and Restated Credit Agreement",
                date=_doc_date(text), type="amendment", amendment_number=n, filing_url=url,
            )
        else:
            doc = Document(
                id=DOC_BASE, name="Third Amended and Restated Credit Agreement",
                date=_doc_date(text), type="base", filing_url=url,
            )
        found.append((doc, path, is_ocr))
    found.sort(key=lambda t: (t[0].type != "base", t[0].amendment_number or 0))
    return {doc.id: (doc, path, is_ocr) for doc, path, is_ocr in found}
ARTICLE_RE = re.compile(r"^(?:ARTICLE|Article)\s+([IVX]+)\b\s*(.*)$")
SECTION_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\s+(.*)$", re.S)
SUBSEC_RE = re.compile(r"^\(([a-z]{1,2})\)\s*(.*)$", re.S)


def read_html(data_dir: Path, doc_id: str) -> str:
    _, path, _ = discover_docs(data_dir)[doc_id]
    return path.read_text(encoding="windows-1252")


def _leaf_paragraphs(html: str) -> list[tuple[str, bool, bool]]:
    """(text, starts_bold, is_table) in document order.

    Leaf divs outside tables become paragraphs; each top-level table becomes one
    paragraph so content tables (pricing grids, schedules) stay with their section.
    """
    soup = BeautifulSoup(html, "lxml")
    paras = []
    for el in soup.find_all(["div", "table"]):
        if el.name == "table":
            if el.find_parent("table"):
                continue
            txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True).replace("\xa0", " "))
            if txt:
                paras.append((txt, False, True))
            continue
        if el.find("div") or el.find_parent("table"):
            continue
        txt = el.get_text(" ", strip=True).replace("\xa0", " ")
        txt = re.sub(r" {2,}", " ", txt)
        if not txt:
            continue
        first_font = next((f for f in el.find_all("font") if f.get_text(strip=True)), None)
        bold = bool(first_font and "font-weight:bold" in (first_font.get("style") or ""))
        paras.append((txt, bold, False))
    return paras


ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def parse_machine_html(html: str, doc_id: str) -> list[Section]:
    """Base agreement + First Amendment: classify leaf paragraphs into a section tree.

    Subsections split only on sequential letters ((a) then (b)...) so lettered lists
    inside definitions or nested clauses stay as body text. §1.01 (Defined Terms) is
    never split — definitions become their own nodes via the extractor.
    """
    sections: list[Section] = []
    cur_article = cur_section = cur_subsec = None
    n_subsecs = 0

    def flush(sec: Section | None, buf: list[str]) -> None:
        if sec is not None:
            sec.text = "\n".join(buf).strip()

    buf: list[str] = []
    for txt, bold, is_table in _leaf_paragraphs(html):
        if is_table and cur_article is None:
            continue  # TOC tables precede the body
        art_m = ARTICLE_RE.match(txt) if bold else None
        sec_m = SECTION_RE.match(txt) if bold else None
        sub_m = SUBSEC_RE.match(txt)
        expected = ALPHABET[n_subsecs] if n_subsecs < 26 else None
        splittable = cur_section is not None and cur_section.title[:13] != "Defined Terms"
        if sub_m and not (splittable and sub_m.group(1) == expected):
            sub_m = None
        if art_m and art_m.group(1) in ROMAN:
            if cur_article is None and buf:
                sections.append(Section(
                    id=f"{doc_id}::preamble", number="0", title="Preamble",
                    text="\n".join(buf).strip(), document_id=doc_id, level="section",
                ))
            flush(cur_subsec or cur_section or cur_article, buf)
            buf = []
            num = art_m.group(1)
            cur_article = Section(
                id=f"{doc_id}::art{ROMAN[num]}", number=num,
                title=art_m.group(2).strip(), text="", document_id=doc_id, level="article",
            )
            sections.append(cur_article)
            cur_section = cur_subsec = None
            n_subsecs = 0
        elif sec_m and cur_article is not None:
            flush(cur_subsec or cur_section, buf)
            num, rest = sec_m.group(1), sec_m.group(2)
            title = rest.split(".")[0].strip() if rest else ""
            buf = [rest.strip()]
            cur_section = Section(
                id=f"{doc_id}::{num}", number=num, title=title[:120],
                text="", document_id=doc_id, level="section",
            )
            sections.append(cur_section)
            cur_subsec = None
            n_subsecs = 0
        elif sub_m and cur_section is not None:
            flush(cur_subsec or cur_section, buf)
            letter = sub_m.group(1)
            num = f"{cur_section.number}({letter})"
            buf = [txt]
            cur_subsec = Section(
                id=f"{doc_id}::{num}", number=num, title="",
                text="", document_id=doc_id, level="subsection",
            )
            sections.append(cur_subsec)
            n_subsecs += 1
        else:
            buf.append(txt)
    flush(cur_subsec or cur_section or cur_article, buf)
    return sections


def extract_ocr_text(html: str) -> str:
    """Second Amendment: pull OCR text hidden in white 1pt font blocks, clean artifacts."""
    blocks = re.findall(r"<font[^>]*color:white[^>]*>(.*?)</font>", html, re.S)
    from html import unescape
    text = " ".join(unescape(re.sub(r"<[^>]+>", " ", b)) for b in blocks)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


OCR_HEADING_RE = re.compile(
    r"(?<!Section )(?<!Sections )(?<!clause )(?<!Article )(\d{1,2}\.\d{1,2})\s+(?=[A-Z\u201c(])"
)
# Amendment Article I headers often appear out of numeric order in OCR streams
# (e.g. 1.11–1.21 after a jumped §2.13/§3.03). Always accept these.
OCR_AMEND_OP_HEADING_RE = re.compile(
    r"(\d{1,2}\.\d{1,2})\s+Amendment to\b"
)


def parse_ocr_html(html: str, doc_id: str) -> list[Section]:
    """OCR white-font text → sections. Sequence-consistent headings plus always-on
    'N.N Amendment to …' markers so later Article I ops are not swallowed."""
    text = extract_ocr_text(html)
    marks: list[tuple[int, str | None, str | None]] = [(0, None, "PREAMBLE")]
    seen_starts: set[int] = {0}
    cur_major, cur_minor = 0, 0
    for m in OCR_HEADING_RE.finditer(text):
        major, minor = (int(x) for x in m.group(1).split("."))
        if (major == cur_major and minor > cur_minor) or (major == cur_major + 1):
            if m.start() not in seen_starts:
                marks.append((m.start(), m.group(1), None))
                seen_starts.add(m.start())
            cur_major, cur_minor = major, minor
    for m in OCR_AMEND_OP_HEADING_RE.finditer(text):
        if m.start() not in seen_starts:
            marks.append((m.start(), m.group(1), None))
            seen_starts.add(m.start())
    marks.sort(key=lambda t: t[0])
    sections = []
    for i, (start, num, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[start:end].strip()
        sec_id = f"{doc_id}::{num or 'preamble'}"
        # Prefer "Amendment to …" title when present
        am = re.match(r"\d+\.\d+\s+(Amendment to\b.{0,100})", body)
        if am:
            title = am.group(1).strip()[:120]
        elif num and "." in body:
            title = body.split(".", 1)[1].strip()[:120]
        else:
            title = label or ""
        sections.append(Section(
            id=sec_id, number=num or "0", title=title, text=body,
            document_id=doc_id, level="section", source="ocr",
        ))
    return sections


def parse_all(data_dir: Path) -> tuple[dict[str, list[Section]], dict[str, Document]]:
    """Parse every discovered document. Returns (sections by doc_id, Document by doc_id)."""
    discovered = discover_docs(data_dir)
    out: dict[str, list[Section]] = {}
    for doc_id, (_, path, is_ocr) in discovered.items():
        html = path.read_text(encoding="windows-1252")
        parse = parse_ocr_html if is_ocr else parse_machine_html
        out[doc_id] = parse(html, doc_id)
        logger.info("%s: %d sections (%s)", doc_id, len(out[doc_id]), "ocr" if is_ocr else "html")
    return out, {doc_id: doc for doc_id, (doc, _, _) in discovered.items()}
