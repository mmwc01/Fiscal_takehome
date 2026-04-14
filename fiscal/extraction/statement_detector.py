"""
Statement detector: identifies which financial statement a PDF page contains.

Input:  raw page text (str) from pdfplumber or any source
Output: DetectionResult with statement_type, reason, and confidence

Detection strategy
------------------
Two tiers, applied in order:

1. PAGE_TRIGGERS — title-level phrases that appear in section headings
   ("Income Statement", "Statement of Cash Flows", etc.)
   A match here means the page heading explicitly names the statement.
   → confidence: "high"

2. TABLE_ANCHORS — content-level terms that appear inside the statement body
   ("Total assets", "Operating activities", "Gross profit", etc.)
   Used only when no page trigger matched.
   → confidence: "medium"

3. No match → statement_type "unknown", confidence "low"

This version is intentionally a bit more recall-friendly than the previous one.
The extraction layer still performs table-shape checks and downstream filtering.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionResult:
    statement_type: str  # income_statement | balance_sheet | cash_flow_statement | unknown
    reason: str
    confidence: str      # high | medium | low


_UNKNOWN = DetectionResult(
    statement_type="unknown",
    reason="no keywords matched",
    confidence="low",
)


PAGE_TRIGGERS: dict[str, list[re.Pattern]] = {
    "income_statement": [
        re.compile(r"\bincome\s+statement\b", re.IGNORECASE),
        re.compile(r"\bprofit\s+(?:and|&)\s+loss\b", re.IGNORECASE),
        re.compile(r"\bstatement\s+of\s+(?:comprehensive\s+)?(?:profit|income)\b", re.IGNORECASE),
        re.compile(r"\bconsolidated\s+(?:income|profit)\b", re.IGNORECASE),
        re.compile(r"\bresults?\s+from\s+operations?\b", re.IGNORECASE),
    ],
    "balance_sheet": [
        re.compile(r"\bbalance\s+sheet\b", re.IGNORECASE),
        re.compile(r"\bstatements?\s+of\s+financial\s+position\b", re.IGNORECASE),
        re.compile(r"\bconsolidated\s+balance\b", re.IGNORECASE),
    ],
    "cash_flow_statement": [
        re.compile(r"\bcash\s+flow\s+statement\b", re.IGNORECASE),
        re.compile(r"\bstatement\s+of\s+cash\s+flows?\b", re.IGNORECASE),
        re.compile(r"\bcash\s+flows?\s+from\b", re.IGNORECASE),
    ],
}


TABLE_ANCHORS: dict[str, list[re.Pattern]] = {
    "income_statement": [
        re.compile(r"\brevenue\b", re.IGNORECASE),
        re.compile(r"\bnet\s+revenue\b", re.IGNORECASE),
        re.compile(r"\bgross\s+profit\b", re.IGNORECASE),
        re.compile(r"\bebitda?\b", re.IGNORECASE),
        re.compile(r"\boperating\s+(?:profit|income|result)\b", re.IGNORECASE),
        re.compile(r"\bnet\s+(?:income|profit|loss)\b", re.IGNORECASE),
        re.compile(r"\bprofit\s+before\s+tax\b", re.IGNORECASE),
        re.compile(r"\bincome\s+tax(?:es)?\b", re.IGNORECASE),
    ],
    "balance_sheet": [
        re.compile(r"\btotal\s+assets\b", re.IGNORECASE),
        re.compile(r"\btotal\s+equity\b", re.IGNORECASE),
        re.compile(r"\btotal\s+liabilities\b", re.IGNORECASE),
        re.compile(r"\bcurrent\s+assets\b", re.IGNORECASE),
        re.compile(r"\bcurrent\s+liabilities\b", re.IGNORECASE),
        re.compile(r"\bcash\s+and\s+cash\s+equivalents\b", re.IGNORECASE),
        re.compile(r"\btrade\s+receivables\b", re.IGNORECASE),
        re.compile(r"\bshareholders['\u2019]?\s+equity\b", re.IGNORECASE),
    ],
    "cash_flow_statement": [
        re.compile(r"\boperating\s+activities\b", re.IGNORECASE),
        re.compile(r"\binvesting\s+activities\b", re.IGNORECASE),
        re.compile(r"\bfinancing\s+activities\b", re.IGNORECASE),
        re.compile(r"\bnet\s+cash\b", re.IGNORECASE),
        re.compile(r"\bcash\s+generated\s+from\s+operations\b", re.IGNORECASE),
        re.compile(r"\bopening\s+cash\s+position\b", re.IGNORECASE),
        re.compile(r"\bclosing\s+cash\s+position\b", re.IGNORECASE),
    ],
}

# Relaxed from 2 -> 1 to improve recall on sparse / oddly formatted pages.
_MIN_ANCHOR_MATCHES = 1

_NOTES_PAGE_RE = re.compile(
    r"\bnotes?\s+to\s+the\b"
    r"|\baccounting\s+polic(?:y|ies)\b"
    r"|\bnote\s+\d+\s*[:\-–]"
    r"|\blong[\-\s]term\s+incentive\b"
    r"|\bshare[\-\s]based\s+(?:payment|compensation|award)\b"
    r"|\bvesting\s+(?:period|condition|date)\b"
    r"|\bperformance\s+(?:share|unit|condition)\b",
    re.IGNORECASE,
)

_NOTES_HEADING_RE = re.compile(
    r"\bnotes?\s+to\s+the\s+(?:consolidated\s+)?"
    r"(?:income\s+statement|profit\s+(?:and|&)\s+loss|"
    r"balance\s+sheet|statement\s+of\s+financial\s+position|"
    r"cash\s+flow\s+statement|statement\s+of\s+cash\s+flows?)\b",
    re.IGNORECASE,
)

_PROSE_STMT_REF_RE = re.compile(
    r"(?:our|in\s+(?:the|our)|of\s+the|classified\s+in|presentation\s+of|"
    r"within\s+the?|affect\s+the)\s+"
    r"(?:consolidated\s+)?(?:income\s+statements?|profit\s+(?:and|&)\s+loss|"
    r"balance\s+sheet|cash\s+flow\s+statements?)\b",
    re.IGNORECASE,
)

_MAX_HEADING_OFFSET = 700


def detect(page_text: str) -> DetectionResult:
    if not page_text or not page_text.strip():
        return _UNKNOWN

    trigger_hits = _find_matches(PAGE_TRIGGERS, page_text)

    if trigger_hits:
        if _NOTES_HEADING_RE.search(page_text):
            return _UNKNOWN

        if not _has_genuine_heading(page_text):
            return _UNKNOWN

        best, matched_text = _best_match(trigger_hits)
        reason = f"page trigger matched: {matched_text!r}"
        if len(trigger_hits) > 1:
            others = ", ".join(k for k in sorted(trigger_hits) if k != best)
            reason += f" (ambiguous — also triggered: {others})"
        return DetectionResult(
            statement_type=best,
            reason=reason,
            confidence="high",
        )

    # Suppress obvious note pages, but otherwise allow anchor-based recall.
    if _NOTES_PAGE_RE.search(page_text):
        return _UNKNOWN

    anchor_hits = _find_matches(TABLE_ANCHORS, page_text)
    strong = {t: hits for t, hits in anchor_hits.items() if len(hits) >= _MIN_ANCHOR_MATCHES}

    if strong:
        best, _ = _best_match(strong)
        terms_preview = ", ".join(f"{h!r}" for h in strong[best][:3])
        return DetectionResult(
            statement_type=best,
            reason=f"anchor terms matched (no page trigger): {terms_preview}",
            confidence="medium",
        )

    return _UNKNOWN


def _find_matches(
    patterns: dict[str, list[re.Pattern]],
    text: str,
) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for stmt_type, compiled in patterns.items():
        matched = [
            m.group(0)
            for p in compiled
            if (m := p.search(text)) is not None
        ]
        if matched:
            hits[stmt_type] = matched
    return hits


def _has_genuine_heading(page_text: str) -> bool:
    for patterns in PAGE_TRIGGERS.values():
        for pattern in patterns:
            m = pattern.search(page_text)
            if m is None:
                continue
            if m.start() > _MAX_HEADING_OFFSET:
                continue
            window_start = max(0, m.start() - 60)
            window = page_text[window_start: m.end() + 40]
            if not _PROSE_STMT_REF_RE.search(window):
                return True
    return False


def _best_match(hits: dict[str, list[str]]) -> tuple[str, str]:
    best = max(hits, key=lambda t: (len(hits[t]), t))
    return best, hits[best][0]