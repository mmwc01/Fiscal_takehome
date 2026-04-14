"""
LLM-based field value extraction — per-statement combined extract+align.

For each PDF, THREE parallel LLM calls (one per statement type) each receive:
  - The catalog fields for that statement type only
  - Both text and table views of the relevant pages (+ adjacent continuation pages)

Smaller per-call prompts reduce truncation risk. All calls across all PDFs run
in a single flattened thread pool — maximum parallelism.

PDF page content (text + tables) is cached to disk by SHA-256 hash so reruns
skip pdfplumber entirely for unchanged PDFs.

Total LLM calls for a 10-year review: N×3 (all parallel)

Entry point: run_field_extraction(company, catalog_path, pdf_dir, client, output_path)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber

from fiscal.extraction import statement_detector
from fiscal.llm.client import LLMClient

logger = logging.getLogger(__name__)

_MIN_PAGE_ALPHA = 100
_MAX_TABLE_ROWS = 150           # cap rows per table to keep prompts manageable
_ADJACENT_PAGES = 2             # continuation pages to include after each HIGH-confidence page
_MAX_PROMPT_CHARS = 200_000     # ~50K tokens — hard cap per LLM call, chunks above this
_STATEMENT_TYPES = ("income_statement", "balance_sheet", "cash_flow_statement")
_STATEMENT_LABEL = {
    "income_statement": "Income Statement",
    "balance_sheet": "Balance Sheet",
    "cash_flow_statement": "Cash Flow Statement",
}
_CACHE_DIR = Path("data/pdf_cache")

# ---------------------------------------------------------------------------
# Per-statement extraction + alignment prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You are a financial data extractor. You receive pages from ONE financial statement section
of an annual report in TWO formats — raw text and structured table rows — both from the
same pages. Use both views and cross-reference them when one is unclear.

Your task: for each catalog field listed, find its value for every year visible in the tables.
Apply financial domain knowledge to match catalog names to document rows — e.g. "Total Revenue"
may appear as "Net revenue" or "Revenue from contracts with customers" depending on the company.

Return JSON where keys are catalog field names and values are year→value dicts:
{
  "Catalog Field Name": {"2024": 1234.0, "2023": 5678.0},
  "Another Field":      {"2024": 900.0,  "2023": 800.0}
}

Rules:
- Return your best match for each field. If a row clearly corresponds to a catalog field
  (even with different wording), include it. Only omit fields genuinely absent from the document.
- Amounts in parentheses (1,234) are negative: -1234.
- Values are plain numbers as they appear (do not scale or convert units).
- Year keys are strings ("2024"). Values are plain numbers (never null).
- Do not include fields not in the catalog list.
""".strip()


# ---------------------------------------------------------------------------
# PDF cache helpers
# ---------------------------------------------------------------------------

def _pdf_hash(pdf_path: Path) -> str:
    """SHA-256 of PDF file contents — used as cache key."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_tables(tables: list) -> str:
    """Render pdfplumber tables as compact markdown rows."""
    parts: list[str] = []
    for ti, table in enumerate(tables or [], 1):
        if not table:
            continue
        rows = []
        for row in table[:_MAX_TABLE_ROWS]:
            cells = [str(c or "").strip() for c in row]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            parts.append(f"Table {ti}:\n" + "\n".join(rows))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# PDF page classification + dual content extraction (with cache)
# ---------------------------------------------------------------------------

def _classify_pdf_pages(
    pdf_path: Path,
    cache_dir: Path,
) -> dict[str, list[dict]]:
    """
    Classify pages and return both text and table content grouped by statement type.
    Each page entry: {"page_number": int, "text": str, "tables_md": str}

    Adjacent continuation pages (up to _ADJACENT_PAGES) are included after each
    detected page, provided they don't begin a different statement type.

    Results are cached by PDF SHA-256 hash — reruns skip pdfplumber for unchanged PDFs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{pdf_path.stem}.json"
    cache_key = _pdf_hash(pdf_path)

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key:
                logger.info("[extractor] %s — cache hit", pdf_path.name)
                return cached["pages_by_statement"]
        except Exception:
            pass

    # First pass: extract content + classify every qualifying page
    classified: dict[int, tuple[str, str]] = {}  # page_number → (stmt_type, confidence)
    all_pages: dict[int, dict] = {}               # page_number → page content dict

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if not text.strip() or sum(1 for c in text if c.isalpha()) < _MIN_PAGE_ALPHA:
                continue
            tables_md = _format_tables(page.extract_tables() or [])
            all_pages[page.page_number] = {
                "page_number": page.page_number,
                "text": text,
                "tables_md": tables_md,
            }
            det = statement_detector.detect(text)
            if det.statement_type in _STATEMENT_TYPES and det.confidence in ("high", "medium"):
                classified[page.page_number] = (det.statement_type, det.confidence)

    # Second pass: HIGH-confidence pages as anchors; expand to adjacent continuation pages.
    # Medium-confidence pages scattered through narrative sections are NOT anchors —
    # they're only included if they happen to be adjacent to a high-confidence page.
    pages: dict[str, list[dict]] = {s: [] for s in _STATEMENT_TYPES}
    included: set[int] = set()

    for page_num in sorted(classified):
        stmt, confidence = classified[page_num]

        # Only high-confidence pages seed the expansion
        if confidence != "high":
            continue

        if page_num not in included:
            pages[stmt].append(all_pages[page_num])
            included.add(page_num)

        for offset in range(1, _ADJACENT_PAGES + 1):
            neighbor = page_num + offset
            if neighbor in included or neighbor not in all_pages:
                continue
            # Stop if the neighbor is classified as a *different* statement type
            neighbor_class = classified.get(neighbor)
            if neighbor_class is not None and neighbor_class[0] != stmt:
                break
            pages[stmt].append(all_pages[neighbor])
            included.add(neighbor)

    try:
        cache_file.write_text(
            json.dumps(
                {"cache_key": cache_key, "pages_by_statement": pages},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("[extractor] cache write failed %s: %s", pdf_path.name, exc)

    for stmt in _STATEMENT_TYPES:
        logger.info(
            "[extractor] %s — %s: %d pages (incl. continuations)",
            pdf_path.name, stmt, len(pages[stmt]),
        )

    return pages


# ---------------------------------------------------------------------------
# Per-statement extraction + alignment (one LLM call per statement per PDF)
# ---------------------------------------------------------------------------

def _build_user_prompt(
    pages: list[dict],
    stmt: str,
    catalog_labels: list[str],
    report_year: int,
    company: str,
) -> str:
    stmt_label = _STATEMENT_LABEL[stmt]
    catalog_line = f"CATALOG FIELDS ({stmt_label}):\n{json.dumps(catalog_labels)}"
    page_parts = [f"--- {stmt_label.upper()} PAGES ---"]
    for p in pages:
        page_parts.append(f"\n=== Page {p['page_number']} ===")
        page_parts.append("[TEXT]\n" + p["text"])
        if p["tables_md"]:
            page_parts.append("[TABLES]\n" + p["tables_md"])
    return (
        f"Company: {company}\nReport year: {report_year}\n\n"
        + catalog_line + "\n\n"
        + "\n".join(page_parts)
    )


def _parse_llm_response(
    data: dict,
    catalog_labels: list[str],
    stmt: str,
    report_year: int,
) -> dict[tuple, tuple[float, int]]:
    valid = set(catalog_labels)
    result: dict[tuple, tuple[float, int]] = {}
    for cat_label, year_vals in data.items():
        if cat_label not in valid or not isinstance(year_vals, dict):
            continue
        for y_str, value in year_vals.items():
            try:
                period_year = int(y_str)
                val = float(value)
                if 2000 <= period_year <= 2100:
                    key = (stmt, cat_label, period_year)
                    if key not in result:
                        result[key] = (val, report_year)
            except (ValueError, TypeError):
                continue
    return result


def extract_and_align(
    *,
    pages: list[dict],
    stmt: str,
    catalog_labels: list[str],
    report_year: int,
    company: str,
    client: LLMClient,
) -> dict[tuple, tuple[float, int]]:
    """
    One (or more) LLM calls for one statement type in one PDF.
    If the prompt exceeds _MAX_PROMPT_CHARS, pages are split into chunks and
    each chunk is called separately — results are merged, first value wins.

    Returns {(stmt, normalized_label, period_year): (value, report_year)}
    """
    if not pages or not catalog_labels:
        return {}

    # Split pages into chunks that each fit within the prompt limit
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_chars = 0
    header_chars = len(
        f"Company: {company}\nReport year: {report_year}\n\n"
        f"CATALOG FIELDS ({_STATEMENT_LABEL[stmt]}):\n{json.dumps(catalog_labels)}\n\n"
    )

    for p in pages:
        page_chars = len(p["text"]) + len(p.get("tables_md", "")) + 50
        if current_chunk and (header_chars + current_chars + page_chars) > _MAX_PROMPT_CHARS:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        current_chunk.append(p)
        current_chars += page_chars

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) > 1:
        logger.info(
            "[extractor] %s/%s/%s — prompt too large, splitting into %d chunks",
            company, report_year, stmt, len(chunks),
        )

    result: dict[tuple, tuple[float, int]] = {}

    for chunk_idx, chunk_pages in enumerate(chunks):
        user_prompt = _build_user_prompt(chunk_pages, stmt, catalog_labels, report_year, company)
        logger.info(
            "[extractor] %s/%s/%s — LLM call %d/%d (~%d chars)",
            company, report_year, stmt, chunk_idx + 1, len(chunks), len(user_prompt),
        )

        try:
            response = client.complete_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(
                "[extractor] %s/%s/%s — LLM error (chunk %d): %s",
                company, report_year, stmt, chunk_idx + 1, exc,
            )
            continue

        data = response.content if isinstance(response.content, dict) else {}
        chunk_result = _parse_llm_response(data, catalog_labels, stmt, report_year)

        # Merge: first value seen for each (field, year) wins
        for key, val in chunk_result.items():
            if key not in result:
                result[key] = val

    logger.info(
        "[extractor] %s/%s/%s — %d fields filled", company, report_year, stmt, len(result)
    )
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_field_extraction(
    company: str,
    *,
    catalog_path: Path,
    pdf_dir: Path,
    client: LLMClient,
    output_path: Path,
    max_pdfs: int = 10,
) -> Path:
    """
    Per-statement extract+align pipeline for one company.

    Phase 1 (parallel): Classify every PDF's pages (text + tables + continuations),
                        cache results to disk.
    Phase 2 (parallel): Fire all LLM calls in a single flattened pool —
                        N PDFs × 3 statement types, all concurrent.
    Phase 3 (Python):   Merge across PDFs (same-year report is authoritative).
    """
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_year = catalog.get("report_year")

    catalog_fields: dict[str, list[dict]] = {}
    for stmt in _STATEMENT_TYPES:
        catalog_fields[stmt] = [
            {"raw_label": f["raw_label"], "normalized_label": f["normalized_label"]}
            for f in catalog.get("fields", {}).get(stmt, [])
            if f.get("raw_label") and f.get("normalized_label")
        ]

    total_catalog = sum(len(v) for v in catalog_fields.values())
    logger.info(
        "[extractor] %s — catalog: %d fields (IS=%d BS=%d CF=%d)",
        company, total_catalog,
        len(catalog_fields["income_statement"]),
        len(catalog_fields["balance_sheet"]),
        len(catalog_fields["cash_flow_statement"]),
    )

    pdfs = sorted(
        pdf_dir.glob("*.pdf"),
        key=lambda p: int(m.group(1)) if (m := re.search(r"(20\d{2})", p.name)) else 0,
        reverse=True,
    )[:max_pdfs]

    if not pdfs:
        raise RuntimeError(f"No PDFs found in {pdf_dir}")

    cache_dir = _CACHE_DIR / company

    # ── Phase 1: classify all PDFs in parallel ────────────────────────────────
    def _classify_one(pdf_path: Path) -> tuple[Path, int, dict]:
        m = re.search(r"(20\d{2})", pdf_path.name)
        ry = int(m.group(1)) if m else 0
        logger.info("[extractor] %s — classifying %s", company, pdf_path.name)
        return pdf_path, ry, _classify_pdf_pages(pdf_path, cache_dir)

    classified: list[tuple[Path, int, dict]] = []
    with ThreadPoolExecutor(max_workers=min(len(pdfs), 8)) as pool:
        futs = {pool.submit(_classify_one, p): p for p in pdfs}
        for f in as_completed(futs):
            try:
                classified.append(f.result())
            except Exception as exc:
                logger.warning(
                    "[extractor] %s — classify failed %s: %s", company, futs[f].name, exc
                )

    logger.info(
        "[extractor] %s — classified %d PDFs, firing LLM calls", company, len(classified)
    )

    # ── Phase 2: flattened (pdf × stmt) LLM calls — all parallel ─────────────
    # Each task: one statement type for one PDF
    Task = tuple[Path, int, str, list[dict], list[str]]
    tasks: list[Task] = []
    for pdf_path, report_year, pages_by_stmt in classified:
        for stmt in _STATEMENT_TYPES:
            pages = pages_by_stmt.get(stmt, [])
            labels = [f["normalized_label"] for f in catalog_fields.get(stmt, [])]
            if pages and labels:
                tasks.append((pdf_path, report_year, stmt, pages, labels))

    best: dict[tuple, tuple[float, int]] = {}

    def _run_task(task: Task) -> tuple[int, dict]:
        pdf_path, report_year, stmt, pages, labels = task
        return report_year, extract_and_align(
            pages=pages,
            stmt=stmt,
            catalog_labels=labels,
            report_year=report_year,
            company=company,
            client=client,
        )

    if not tasks:
        logger.warning("[extractor] %s — no tasks (no pages detected in any PDF)", company)

    if tasks:
        with ThreadPoolExecutor(max_workers=min(len(tasks), 16)) as pool:
            futs = {pool.submit(_run_task, t): t for t in tasks}
            for f in as_completed(futs):
                task = futs[f]
                try:
                    report_year, aligned = f.result()
                except Exception as exc:
                    logger.warning(
                        "[extractor] %s — task failed %s/%s: %s",
                        company, task[1], task[2], exc,
                    )
                    continue

                # ── Phase 3: merge (newest report wins for every period) ────────
                # The 2024 report's 2023 comparative column is a restatement
                # and more authoritative than the 2023 report's own 2023 column.
                # So for any given (stmt, field, period_year), we keep the value
                # from whichever report has the highest report year.
                for key, (value, src_year) in aligned.items():
                    existing = best.get(key)
                    if existing is None:
                        best[key] = (value, src_year)
                    else:
                        _, prev_src = existing
                        if src_year > prev_src:
                            best[key] = (value, src_year)

    # Build output preserving catalog field order
    output_statements: dict[str, list[dict]] = {}
    for stmt in _STATEMENT_TYPES:
        rows = []
        for f in catalog_fields[stmt]:
            norm = f["normalized_label"]
            year_vals = {
                str(py): v
                for (s, nl, py), (v, _) in best.items()
                if s == stmt and nl == norm
            }
            rows.append({
                "raw_label": f["raw_label"],
                "normalized_label": norm,
                "values_by_year": dict(
                    sorted(year_vals.items(), key=lambda x: int(x[0]), reverse=True)
                ),
            })
        output_statements[stmt] = rows

    payload: dict[str, Any] = {
        "company": company,
        "catalog_year": catalog_year,
        "currency": catalog.get("currency"),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pdfs_processed": [p.name for p in pdfs],
        "statements": output_statements,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[extractor] %s — saved to %s", company, output_path)
    return output_path
