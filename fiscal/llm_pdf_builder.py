"""
PDF report builder for LLM-discovered fields.

Reads a company_field_values JSON produced by field_value_extractor, runs an LLM
classification pass to deduplicate and structure the rows into a clean hierarchy
(section headers, line items, subtotals, key metrics), then renders a landscape A4
PDF with one table per statement type.

Entry point: build_pdf_from_values(values_path, output_dir)
"""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logger = logging.getLogger(__name__)

MAX_YEARS = 10
MIN_POPULATED_YEARS = 2

SECTION_ORDER = ["income_statement", "balance_sheet", "cash_flow_statement"]
SECTION_LABEL = {
    "income_statement": "Income Statement",
    "balance_sheet": "Balance Sheet",
    "cash_flow_statement": "Cash Flow Statement",
}

_DEFAULT_YEAR_END = "Dec 31"
_FISCAL_YEAR_END: dict[str, str] = {}

# Colours
_COL_HEADER_BG    = colors.HexColor("#F2F2F2")
_SECTION_HDR_BG   = colors.HexColor("#E8E8E8")
_TABLE_HEADER_BG  = colors.HexColor("#D8D8D8")
_GRAND_TOTAL_BG   = colors.HexColor("#F5F5F5")
_TITLE_COLOR      = colors.HexColor("#1A1A2E")
_GRID_COLOR       = colors.HexColor("#CCCCCC")
_TEXT_COLOR       = colors.black
_DASH_COLOR       = colors.HexColor("#BBBBBB")
_MUTED_COLOR      = colors.HexColor("#888888")

# Row style constants
_STYLE_SECTION_HEADER = "section_header"
_STYLE_LINE_ITEM      = "line_item"
_STYLE_TOTAL          = "total"
_STYLE_GRAND_TOTAL    = "grand_total"


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

_CLASSIFICATION_SYSTEM = """\
You are a financial data expert. You will receive a list of line-item labels
extracted from a company annual report for one statement type, and you must
reorganize them into a clean, standard hierarchy.

Return a JSON object with a single "rows" array. Each element has:
  "label"        — display label (use standard financial terminology)
  "style"        — one of: section_header | line_item | total | grand_total
  "source_labels" — list of input labels that map to this display row

Styles:
  section_header  Bold section title, no data values (e.g. "Revenues", "Cost of Revenues").
                  source_labels must be [].
  line_item       Regular indented row with values.
  total           Subtotal, bold (e.g. "Total Revenues", "Total Operating Expenses").
  grand_total     Key metric, bold + highlighted (e.g. "Gross Profit", "Net Income",
                  "Net Cash from Operating Activities").

Rules:
1. DEDUPLICATE: if multiple input labels mean the same thing, merge them into one
   row — list ALL of them in source_labels (first match will be used for values).
2. Only emit line_item / total / grand_total rows that have at least one
   source_label present in the input list. Never invent rows.
3. section_header rows are purely structural — no source_labels, no values.
4. Preserve logical order: header → its items → subtotal/total → next header.
5. Do not drop rows unless they are true semantic duplicates of another row.
"""

_CLASSIFICATION_TEMPLATES = {
    "income_statement": """\
Statement type: Income Statement
Company: {company}

Input labels (extracted from annual report):
{labels}

Organize using this standard structure (skip any section for which no input
labels exist):
- section_header "Revenues" → line items → total "Total Revenues"
- section_header "Cost of Revenues" → line items → total "Total Cost of Revenues"
- grand_total "Gross Profit"
- section_header "Operating Expenses" → line items → total "Total Operating Expenses"
- grand_total "Income from Operations"
- Other income / expense line items
- grand_total "Income Before Income Taxes"
- line_item tax provision
- grand_total "Net Income"
- Earnings per share section (section_header + line items) if labels exist
- Weighted-average shares section (section_header + line items) if labels exist
""",
    "balance_sheet": """\
Statement type: Balance Sheet
Company: {company}

Input labels (extracted from annual report):
{labels}

Organize using this standard structure (skip any section for which no input
labels exist):
- section_header "Assets"
  - section_header "Current Assets" → line items → total "Total Current Assets"
  - section_header "Non-Current Assets" → line items → total "Total Non-Current Assets"
  - grand_total "Total Assets"
- section_header "Liabilities"
  - section_header "Current Liabilities" → line items → total "Total Current Liabilities"
  - section_header "Non-Current Liabilities" → line items → total "Total Non-Current Liabilities"
  - total "Total Liabilities"
- section_header "Equity" → line items → total "Total Equity"
- grand_total "Total Liabilities and Equity"
""",
    "cash_flow_statement": """\
Statement type: Cash Flow Statement
Company: {company}

Input labels (extracted from annual report):
{labels}

Organize using this standard structure (skip any section for which no input
labels exist):
- section_header "Operating Activities" → line items → grand_total "Net Cash from Operating Activities"
- section_header "Investing Activities" → line items → grand_total "Net Cash from Investing Activities"
- section_header "Financing Activities" → line items → grand_total "Net Cash from Financing Activities"
- grand_total "Net Change in Cash" if present
- Opening / Closing cash balance line items if present
""",
}


def _classify_with_llm(
    stmt: str,
    pivoted_rows: list[dict],
    company: str,
    client,
) -> list[dict] | None:
    """
    Call the LLM to deduplicate and classify pivoted rows into a hierarchy.
    Returns a list of dicts with keys: label, style, source_labels.
    Returns None on failure (caller falls back to flat list).
    """
    labels_block = "\n".join(f"- {r['label']}" for r in pivoted_rows)
    template = _CLASSIFICATION_TEMPLATES.get(stmt)
    if not template:
        return None

    user_prompt = template.format(company=company, labels=labels_block)

    try:
        resp = client.complete_json(
            system_prompt=_CLASSIFICATION_SYSTEM,
            user_prompt=user_prompt,
        )
        rows = resp.content.get("rows", [])
        if not isinstance(rows, list) or not rows:
            return None
        return rows
    except Exception as exc:
        logger.warning("[pdf_builder] classification LLM failed (%s): %s", stmt, exc)
        return None


def _apply_classification(
    classified: list[dict],
    pivoted_rows: list[dict],
    years: list[int],
) -> list[dict]:
    """
    Merge LLM classification output with pivoted value rows.
    Returns display-ready rows with style, label, and year values.
    Each source label can only be consumed once (prevents duplicate rows).
    """
    label_to_data: dict[str, dict] = {r["label"]: r for r in pivoted_rows}
    used_sources: set[str] = set()
    result: list[dict] = []

    for item in classified:
        style = item.get("style", _STYLE_LINE_ITEM)
        label = (item.get("label") or "").strip()
        source_labels: list[str] = item.get("source_labels") or []

        if not label:
            continue

        if style == _STYLE_SECTION_HEADER:
            result.append({"label": label, "style": style, **{y: None for y in years}})
            continue

        # Find the first unconsumed source_label that has data
        data_row: dict | None = None
        matched_source: str | None = None
        for sl in source_labels:
            if sl in label_to_data and sl not in used_sources:
                data_row = label_to_data[sl]
                matched_source = sl
                break
        # Also try the display label itself if no source matched
        if data_row is None and label in label_to_data and label not in used_sources:
            data_row = label_to_data[label]
            matched_source = label

        if data_row is None:
            continue  # no matching data — skip

        used_sources.add(matched_source)  # type: ignore[arg-type]

        row: dict = {"label": label, "style": style}
        for y in years:
            v = data_row.get(y)
            row[y] = abs(v) if v is not None else None
        result.append(row)

    return result


# ---------------------------------------------------------------------------
# Value plausibility filter
# ---------------------------------------------------------------------------

def _is_plausible(v: float, other_vals: list[float]) -> bool:
    """
    Return False if v looks like an extraction artifact rather than a real value.

    Heuristic: a non-zero value that is less than 0.1% of the median of the
    other populated values in the same row is almost certainly a footnote
    reference or page number that the LLM mistook for a financial figure
    (e.g. the value 1 in a row where every other year is in the millions).

    Genuine zeros are always kept — they represent real "no activity" years.
    Small-magnitude rows (e.g. EPS, share counts in the single digits) are
    unaffected because their median is also small.
    """
    if v == 0.0:
        return True
    if not other_vals:
        return True
    med = statistics.median(other_vals)
    if med > 100 and abs(v) < med * 0.001:
        return False
    return True


# ---------------------------------------------------------------------------
# Plain pivot (fallback — no LLM classification)
# ---------------------------------------------------------------------------

def _pivot_rows(rows: list[dict], years: list[int]) -> list[dict]:
    """Pivot raw extracted rows → flat display rows (no classification)."""
    seen_labels: set[str] = set()
    result: list[dict] = []

    for row in rows:
        label = row.get("raw_label", "")
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)

        values_by_year = row.get("values_by_year") or {}
        raw_vals: dict[int, float] = {}
        for y in years:
            v = values_by_year.get(str(y))
            if v is not None:
                try:
                    raw_vals[y] = abs(float(v))
                except (ValueError, TypeError):
                    pass

        # Filter implausible values (extraction artifacts)
        other_pool = list(raw_vals.values())
        year_vals = {
            y: v for y, v in raw_vals.items()
            if _is_plausible(v, [x for x in other_pool if x != v])
        }

        populated = sum(1 for y in years if year_vals.get(y) is not None)
        if populated < MIN_POPULATED_YEARS:
            continue

        pivoted: dict = {"label": label, "style": _STYLE_LINE_ITEM}
        pivoted.update(year_vals)
        result.append(pivoted)

    return result


def _pivot_rows_for_classification(rows: list[dict], years: list[int]) -> list[dict]:
    """
    Pivot rows and deduplicate by raw_label for the classification step.
    Values are kept as-is (sign preserved) — classification step applies abs().
    """
    seen_labels: set[str] = set()
    result: list[dict] = []

    for row in rows:
        label = row.get("raw_label", "")
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)

        values_by_year = row.get("values_by_year") or {}
        raw_vals: dict[int, float] = {}
        for y in years:
            v = values_by_year.get(str(y))
            if v is not None:
                try:
                    raw_vals[y] = float(v)
                except (ValueError, TypeError):
                    pass

        # Filter implausible values (extraction artifacts)
        other_pool = [abs(x) for x in raw_vals.values()]
        year_vals = {
            y: v for y, v in raw_vals.items()
            if _is_plausible(abs(v), [abs(x) for x in other_pool if x != abs(v)])
        }

        populated = sum(1 for y in years if year_vals.get(y) is not None)
        if populated < MIN_POPULATED_YEARS:
            continue

        pivoted: dict = {"label": label, "style": _STYLE_LINE_ITEM}
        pivoted.update(year_vals)
        result.append(pivoted)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_pdf_from_values(
    values_path: Path,
    output_dir: Path | str = "output",
) -> Path:
    data = json.loads(values_path.read_text(encoding="utf-8"))
    company = data["company"]
    statements = data.get("statements", {})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{company}_llm_report.pdf"

    # Collect all years across sections
    all_years: set[int] = set()
    for stmt in SECTION_ORDER:
        for row in statements.get(stmt, []):
            for y_str in row.get("values_by_year", {}):
                try:
                    all_years.add(int(y_str))
                except ValueError:
                    pass

    years: list[int] = sorted(all_years, reverse=True)[:MAX_YEARS]

    if not years:
        logger.warning("[pdf_builder] %s — no year data found, skipping PDF", company)
        return pdf_path

    # LLM client for classification
    from fiscal.llm.client import LLMClient
    client = LLMClient()

    sections: dict[str, list[dict]] = {}
    for stmt in SECTION_ORDER:
        raw_rows = statements.get(stmt, [])
        if not raw_rows:
            continue

        # Pivot + simple dedup
        pivoted = _pivot_rows_for_classification(raw_rows, years)
        if not pivoted:
            continue

        if client.enabled():
            logger.info("[pdf_builder] %s — classifying %s (%d rows)", company, stmt, len(pivoted))
            classified_meta = _classify_with_llm(stmt, pivoted, company, client)
            if classified_meta:
                display_rows = _apply_classification(classified_meta, pivoted, years)
            else:
                display_rows = _pivot_rows(raw_rows, years)
        else:
            display_rows = _pivot_rows(raw_rows, years)

        if display_rows:
            sections[stmt] = display_rows

    if not sections:
        logger.warning("[pdf_builder] %s — no data to render", company)
        return pdf_path

    _render_pdf(pdf_path, company, sections, years, data)
    logger.info("[pdf_builder] PDF written: %s", pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def _fmt_value(v: float | None) -> str:
    if v is None:
        return "–"
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"


def _col_header(company: str, year: int) -> str:
    month_day = _FISCAL_YEAR_END.get(company, _DEFAULT_YEAR_END)
    return f"{month_day}, {year}"


def _render_pdf(
    path: Path,
    company: str,
    sections: dict[str, list[dict]],
    years: list[int],
    data: dict,
) -> None:
    title_style = ParagraphStyle(
        "FT",
        fontSize=14,
        leading=18,
        textColor=_TITLE_COLOR,
        fontName="Helvetica-Bold",
        spaceAfter=2 * mm,
    )
    sub_style = ParagraphStyle(
        "FS",
        fontSize=8,
        leading=11,
        textColor=_MUTED_COLOR,
        fontName="Helvetica",
        spaceAfter=6 * mm,
    )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"Fiscal Report — {company.title()}",
        author="Fiscal Pipeline",
    )

    # Strip "thousands" / "millions" suffix — just keep the currency code
    raw_currency = data.get("currency") or ""
    currency = raw_currency.split()[0] if raw_currency else ""

    generated = datetime.now(timezone.utc).strftime("%B %d, %Y")
    title_text = f"Financial Report — {company.title()}"
    if currency:
        title_text += f"  ({currency})"
    story = [
        Paragraph(title_text, title_style),
        Paragraph(f"Generated {generated}  ·  Fiscal Pipeline (LLM Strategy)", sub_style),
    ]

    for stmt in SECTION_ORDER:
        display_rows = sections.get(stmt)
        if not display_rows:
            continue
        story.append(Spacer(1, 5 * mm))
        story.append(_build_section_table(display_rows, years, company))

    doc.build(story)


def _build_section_table(
    display_rows: list[dict],
    years: list[int],
    company: str,
) -> Table:
    page_w = landscape(A4)[0] - 24 * mm
    label_col_w = page_w * 0.32
    yr_col_w = (page_w - label_col_w) / max(len(years), 1)
    col_widths = [label_col_w] + [yr_col_w] * len(years)

    # Row 0: column headers only (no section title row)
    col_header_row = [""] + [_col_header(company, y) for y in years]
    table_data = [col_header_row]

    for row in display_rows:
        label = row["label"]
        vals = [_fmt_value(row.get(y)) for y in years]
        table_data.append([label] + vals)

    # --- Base table style ---
    style_cmds = [
        # Column header row (row 0)
        ("BACKGROUND",    (0, 0), (-1, 0), _COL_HEADER_BG),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 6.5),
        ("ALIGN",         (1, 0), (-1, 0), "RIGHT"),
        ("ALIGN",         (0, 0), (0, 0),  "LEFT"),
        ("TOPPADDING",    (0, 0), (-1, 0), 4),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("LINEBELOW",     (0, 0), (-1, 0), 0.75, _GRID_COLOR),

        # Data rows baseline (row 1+)
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 7.5),
        ("ALIGN",         (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
        ("LEFTPADDING",   (0, 1), (0, -1),  12),
        ("TOPPADDING",    (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("TEXTCOLOR",     (0, 1), (-1, -1), _TEXT_COLOR),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
        ("LINEBELOW",     (0, -1), (-1, -1), 0.5, _GRID_COLOR),
    ]

    # --- Per-row style overrides (data starts at row 1 — no section title row) ---
    for ri, row in enumerate(display_rows, start=1):
        row_style = row.get("style", _STYLE_LINE_ITEM)

        if row_style == _STYLE_SECTION_HEADER:
            style_cmds += [
                ("BACKGROUND",    (0, ri), (-1, ri), _SECTION_HDR_BG),
                ("FONTNAME",      (0, ri), (-1, ri), "Helvetica-Bold"),
                ("FONTSIZE",      (0, ri), (-1, ri), 7.5),
                ("LEFTPADDING",   (0, ri), (0, ri),  5),
                ("LINEABOVE",     (0, ri), (-1, ri), 0.5, _GRID_COLOR),
                ("LINEBELOW",     (0, ri), (-1, ri), 0.5, _GRID_COLOR),
                # Suppress values (section headers carry no numbers)
                ("TEXTCOLOR",     (1, ri), (-1, ri), _SECTION_HDR_BG),
            ]

        elif row_style == _STYLE_GRAND_TOTAL:
            style_cmds += [
                ("BACKGROUND",    (0, ri), (-1, ri), _GRAND_TOTAL_BG),
                ("FONTNAME",      (0, ri), (-1, ri), "Helvetica-Bold"),
                ("LEFTPADDING",   (0, ri), (0, ri),  5),
                ("LINEABOVE",     (0, ri), (-1, ri), 0.75, _GRID_COLOR),
                ("LINEBELOW",     (0, ri), (-1, ri), 0.75, _GRID_COLOR),
            ]

        elif row_style == _STYLE_TOTAL:
            style_cmds += [
                ("FONTNAME",      (0, ri), (-1, ri), "Helvetica-Bold"),
                ("LEFTPADDING",   (0, ri), (0, ri),  5),
                ("LINEABOVE",     (0, ri), (-1, ri), 0.4, _GRID_COLOR),
            ]

        # Dim dash cells regardless of row style
        vals = [_fmt_value(row.get(y)) for y in years]
        for ci, v in enumerate(vals, start=1):
            if v == "–":
                style_cmds.append(("TEXTCOLOR", (ci, ri), (ci, ri), _DASH_COLOR))

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl
