"""Prompt templates. Same answer prompt for graph path and baseline — only context differs."""

ANSWER_SYSTEM_BASE = (
    "You are a careful legal analyst answering questions about a credit agreement and its "
    "amendments. Answer ONLY from the provided context. Every claim must cite its source "
    "in square brackets using the citation labels given, e.g. [Credit Agreement \u00a77.05(a), "
    "as amended by Second Amendment (effective 2020-05-13)]. Cite at section or subsection "
    "level. If the context does not contain the answer, say so plainly. If a provision comes "
    "from OCR-extracted text, note that the wording may contain OCR artifacts."
)

# Structured legal reasoning order — experiments/proviso_prompting.py: with this
# addition Qwen-14B fixes Q5 (exception enumeration) and Q10 (amendment proviso).
COT_ADDENDUM = (
    " Reason in this order before answering: (1) Premise check — if the question assumes a"
    " facility, instrument, or fact that the context contradicts, correct the premise first."
    " (2) Baseline rule first — for negative covenants, state the prohibition before any"
    " exception; an action is prohibited unless an express exception covers it."
    " (3) Overrides in force — apply any amendment proviso that suspends or limits earlier"
    " clauses as of the given date, and check whether a cure or grace period actually applies"
    " to the specific section cited rather than assuming a default one."
)

ANSWER_SYSTEM = ANSWER_SYSTEM_BASE + COT_ADDENDUM

ANSWER_PROMPT = """Question{as_of_clause}:
{question}

Context (each block is labelled with its citation):
{context}

Answer the question precisely and concisely, with citations after each claim. If the
question rests on an assumption the documents contradict, correct the assumption. Do not
repeat yourself."""


def build_context(blocks: list[tuple[str, str, str]]) -> str:
    """blocks: (citation, text, source). OCR provenance is surfaced in the label."""
    parts = []
    for citation, text, source in blocks:
        tag = f"[{citation}]" + (" (OCR-extracted text)" if source == "ocr" else "")
        parts.append(f"{tag}\n{text}")
    return "\n\n---\n\n".join(parts)


def build_answer_prompt(question: str, context: str, as_of: str | None) -> str:
    clause = f" (answer as of {as_of})" if as_of else ""
    return ANSWER_PROMPT.format(as_of_clause=clause, question=question, context=context)
