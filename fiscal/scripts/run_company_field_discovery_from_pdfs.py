from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv

from fiscal.extraction import statement_detector
from fiscal.llm.client import LLMClient
from fiscal.llm.company_field_discovery import (
    discover_company_fields,
    save_discovery_result,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Minimum number of alpha characters for a page to be considered non-empty
_MIN_PAGE_ALPHA = 100

# Statement types we want to discover fields for
_STATEMENT_TYPES = ("income_statement", "balance_sheet", "cash_flow_statement")

# Primary statement title must appear at the START of a line within the first
# 300 characters of the page — this avoids cross-references like
# "39. Events after balance sheet date" triggering a false positive.
_PRIMARY_HEADING_RE = re.compile(
    r"^(?:consolidated|combined|company|group)?\s*"
    r"(?:statements?\s+of\s+(?:comprehensive\s+)?(?:income|profit|loss|financial\s+position|cash\s+flows?)"
    r"|(?:income|profit)\s+(?:and\s+loss\s+)?statements?"
    r"|balance\s+sheet"
    r"|cash\s+flow\s+statements?"
    r"|statements?\s+of\s+(?:profit\s+(?:and|&)\s+loss|operations?))",
    re.IGNORECASE | re.MULTILINE,
)
_PRIMARY_HEADING_SEARCH_CHARS = 500


def _extract_year_from_name(path: Path) -> int | None:
    m = re.search(r"(20\d{2})", path.name)
    return int(m.group(1)) if m else None


def _pdfs_newest_first(company: str, base_dir: str | Path = "data/pdfs") -> list[Path]:
    company_dir = Path(base_dir) / company
    if not company_dir.exists():
        raise RuntimeError(f"Company PDF folder not found: {company_dir}")

    pdfs = list(company_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {company_dir}")

    def sort_key(p: Path):
        year = _extract_year_from_name(p)
        return (year if year is not None else -1, p.stat().st_mtime)

    return sorted(pdfs, key=sort_key, reverse=True)


def _has_primary_heading(page_text: str) -> bool:
    """
    Return True if a primary statement title starts a line within the page's
    first 300 characters AND that line doesn't look like a table-of-contents
    entry (i.e. doesn't end with a bare page number like '... 157').
    """
    for m in _PRIMARY_HEADING_RE.finditer(page_text[:_PRIMARY_HEADING_SEARCH_CHARS]):
        # Find the full line containing this match
        line_start = page_text.rfind("\n", 0, m.start()) + 1
        line_end = page_text.find("\n", m.end())
        if line_end == -1:
            line_end = len(page_text)
        line = page_text[line_start:line_end].strip()
        # TOC entries end with dots/spaces then a page number — skip them
        # Require 2+ dots OR 3+ spaces before the number to avoid false positives
        # like "For the year ended December 31" which ends with a legitimate "31"
        if re.search(r"(?:\.{2,}|[ \t]{3,})\s*\d{1,4}\s*$", line):
            continue
        return True
    return False


def _classify_pages(pdf_path: Path) -> dict[str, list[str]]:
    """
    Open the PDF with pdfplumber, extract each page's text, run the
    statement_detector on it, and group pages by detected statement type.

    For discovery we only include pages that:
      1. Are high-confidence detections (not narrative medium-confidence pages).
      2. Have an actual primary statement title in their opening lines.

    This positive filter avoids note/disclosure pages that trigger high-confidence
    detections because they discuss income statement or balance sheet topics.
    """
    pages_by_statement: dict[str, list[str]] = {s: [] for s in _STATEMENT_TYPES}
    pages_scanned = 0
    pages_classified: dict[str, int] = {s: 0 for s in _STATEMENT_TYPES}
    pages_skipped = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        logger.info("Opened PDF: %s (%d pages)", pdf_path.name, len(pdf.pages))

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            if sum(1 for c in page_text if c.isalpha()) < _MIN_PAGE_ALPHA:
                continue

            pages_scanned += 1
            detection = statement_detector.detect(page_text)

            # Only high-confidence hits — medium is usually narrative prose
            if detection.confidence != "high":
                continue

            if detection.statement_type not in _STATEMENT_TYPES:
                continue

            # Must have an actual statement title at the top of the page
            if not _has_primary_heading(page_text):
                pages_skipped += 1
                logger.debug(
                    "  Page %d → skipped (no primary heading, detected as %s)",
                    page.page_number,
                    detection.statement_type,
                )
                continue

            pages_by_statement[detection.statement_type].append(page_text)
            pages_classified[detection.statement_type] += 1
            logger.info(
                "  Page %d → %s: %s",
                page.page_number,
                detection.statement_type,
                detection.reason[:80],
            )

    logger.info(
        "Pages scanned: %d | kept: IS=%d BS=%d CF=%d | skipped (no heading): %d",
        pages_scanned,
        pages_classified["income_statement"],
        pages_classified["balance_sheet"],
        pages_classified["cash_flow_statement"],
        pages_skipped,
    )
    return pages_by_statement


def run(company: str, base_dir: str | Path = "data/pdfs") -> Path:
    client = LLMClient()
    if not client.enabled():
        raise RuntimeError("OPENAI_API_KEY is not set or .env was not loaded")

    pdfs = _pdfs_newest_first(company, base_dir=base_dir)

    # Try PDFs newest-first; use the first one that yields statement pages
    pages_by_statement: dict[str, list[str]] = {}
    pdf_path = pdfs[0]
    for candidate in pdfs:
        logger.info("Trying PDF: %s", candidate.name)
        pages = _classify_pages(candidate)
        total = sum(len(v) for v in pages.values())
        if total > 0:
            pdf_path = candidate
            pages_by_statement = pages
            break
        logger.info("  → no statement pages detected, trying next PDF")
    else:
        logger.warning("No PDF yielded statement pages for %s", company)

    report_year = _extract_year_from_name(pdf_path)
    logger.info("Using PDF: %s (report year: %s)", pdf_path.name, report_year)

    result = discover_company_fields(
        company=company,
        report_year=report_year,
        pages_by_statement=pages_by_statement,
        client=client,
    )

    output_path = save_discovery_result(result)
    logger.info(
        "Discovered %d fields total (IS=%d BS=%d CF=%d)",
        len(result.fields),
        sum(1 for f in result.fields if f.statement_type == "income_statement"),
        sum(1 for f in result.fields if f.statement_type == "balance_sheet"),
        sum(1 for f in result.fields if f.statement_type == "cash_flow_statement"),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate company field catalog from latest PDF in data/pdfs/<company>/"
    )
    parser.add_argument("--company", required=True, help="Company slug, e.g. adyen or sap")
    parser.add_argument("--base-dir", default="data/pdfs", help="Base PDFs directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    path = run(company=args.company, base_dir=args.base_dir)
    print(path)


if __name__ == "__main__":
    main()
