from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fiscal.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-statement system prompts — each call is focused on a single statement
# ---------------------------------------------------------------------------

_SHARED_RULES = """
THOROUGHNESS — this is the most important instruction:
- Extract EVERY row that has numeric values in the table. Do not skip any.
- Include top-level lines, subtotals, section totals, and intermediate calculations.
- Include indented sub-items (e.g. "of which: interest income", "– Depreciation").
- Include both positive and negative line items (costs, losses, deductions).
- When in doubt, include the row. It is far better to include too many rows than to miss any.

What counts as a valid row label:
- Short table-style line item text (noun phrase, not a sentence).
- Appears in a tabular context — adjacent to numeric columns, year headers, or totals.
- Any row in the table that has at least one number next to it.

What to EXCLUDE (only these):
- Narrative commentary or full sentences with verbs ("Revenue increased 12%...").
- Standalone years, note references ("Note 3", "(A.1)"), page numbers.
- Column headers (the year labels themselves, e.g. "2024", "2023").
- Personal names, addresses, auditor firm names.

Evidence rules:
- evidence MUST be a short snippet copied verbatim from the provided page text.
- The snippet must show the raw_label in a table-like context next to numbers.
- Do NOT use evidence from narrative paragraphs or management commentary.
- Keep evidence under 120 characters.

Normalization rules:
- normalized_label must be lowercase.
- Convert "&" to "and", collapse repeated whitespace.
- Preserve the exact business meaning — do not over-generalize.

Deduplication:
- Do not include the same label twice.
- If a label appears with minor formatting variations, keep one best version.

Return ONLY valid JSON and nothing else.
""".strip()


_INCOME_SYSTEM_PROMPT = f"""
You are a financial statement row-label extractor specializing in INCOME STATEMENTS
(also called Profit & Loss statements or Statements of Comprehensive Income).

You will receive text from PDF pages that have been pre-classified as income statement pages.
Your job is to extract EVERY SINGLE row label that appears in the income statement table(s).
Be exhaustive — extract all rows, including subtotals, sub-items, and intermediate lines.

Also detect the reporting currency and unit (e.g. "EUR thousands", "USD millions").

Typical rows you should find (this list is NOT exhaustive — extract everything in the table):
Revenue:
  Net revenue, Total revenue, Gross revenue, Non-interest revenue, Interest income,
  Subscription revenue, Transaction revenue, Service revenue, Licensing revenue,
  Commission income, Net interest income, Premiums earned, Automotive sales,
  Energy generation and storage revenue, Regulatory credits, Other revenue
Costs & gross profit:
  Cost of revenue, Cost of goods sold, Cost of services, Transaction costs,
  Costs incurred from financial institutions, Settlement costs, Claims incurred,
  Gross profit, Gross margin
Operating expenses (extract each sub-line separately):
  Research and development, Sales and marketing, General and administrative,
  Selling general and administrative, Employee benefits, Personnel costs,
  Share-based compensation, Depreciation, Amortization, Depreciation and amortization,
  Impairment, Restructuring, Other operating expenses, Total operating expenses
Operating result:
  Operating income, Operating profit, Operating result, EBIT, EBITDA,
  Income from operations, Results from operations
Non-operating:
  Finance income, Finance costs, Finance expense, Interest expense, Interest income,
  Other income, Other expense, Foreign exchange gains/losses, Share of results of associates
Pre-tax & tax:
  Profit before tax, Income before taxes, Income before income taxes,
  Income tax expense, Tax charge, Current income tax, Deferred income tax,
  Innovation box benefit, Other tax items
Bottom line:
  Net income, Net profit, Profit for the year, Profit attributable to shareholders,
  Non-controlling interests, Profit attributable to non-controlling interests
Other comprehensive income (OCI):
  Currency translation adjustments, Remeasurement of pension obligations,
  Fair value movements, Total other comprehensive income, Total comprehensive income
Per share:
  Basic earnings per share, Diluted earnings per share

Return JSON only with this shape:
{{
  "currency": "e.g. EUR thousands, USD millions, GBP thousands — or null if not found",
  "fields": [
    {{
      "raw_label": "exact label text from the source",
      "normalized_label": "lowercase normalized version",
      "evidence": "short verbatim snippet from table context",
      "sample_value": 1234.5
    }}
  ]
}}

sample_value is REQUIRED — it must be one actual numeric value for this field taken directly
from the table. If you cannot find a real number for a row, do not include that row at all.

{_SHARED_RULES}
""".strip()


_BALANCE_SHEET_SYSTEM_PROMPT = f"""
You are a financial statement row-label extractor specializing in BALANCE SHEETS
(also called Statements of Financial Position).

You will receive text from PDF pages that have been pre-classified as balance sheet pages.
Your job is to extract EVERY SINGLE row label that appears in the balance sheet table(s).
Be exhaustive — extract all rows, including subtotals, sub-items, and section totals.

Also detect the reporting currency and unit (e.g. "EUR thousands", "USD millions").

Typical rows you should find (this list is NOT exhaustive — extract everything in the table):
Current assets:
  Cash and cash equivalents, Cash held at central banks, Cash held at banks,
  Restricted cash, Short-term investments, Marketable securities,
  Trade receivables, Accounts receivable, Other receivables, Contract assets,
  Inventories, Prepaid expenses, Current income tax receivables,
  Other current assets, Total current assets
Non-current assets:
  Property plant and equipment, Right-of-use assets, Lease assets,
  Intangible assets, Goodwill, Capitalized development costs,
  Equity method investments, Financial assets, Investment securities,
  Deferred tax assets, Other non-current assets, Total non-current assets
  Total assets
Current liabilities:
  Trade payables, Accounts payable, Payables to merchants, Other payables,
  Accrued expenses, Short-term borrowings, Current lease liabilities,
  Current portion of long-term debt, Deferred revenue, Contract liabilities,
  Current income tax payables, Other current liabilities, Total current liabilities
Non-current liabilities:
  Long-term debt, Borrowings, Senior notes, Non-current lease liabilities,
  Deferred tax liabilities, Provisions, Pension obligations,
  Other non-current liabilities, Total non-current liabilities
  Total liabilities
Equity:
  Share capital, Common stock, Additional paid-in capital, Share premium,
  Retained earnings, Accumulated other comprehensive income/loss,
  Treasury shares, Other reserves, Non-controlling interests,
  Total equity attributable to shareholders, Total equity
  Total liabilities and equity

Return JSON only with this shape:
{{
  "currency": "e.g. EUR thousands, USD millions, GBP thousands — or null if not found",
  "fields": [
    {{
      "raw_label": "exact label text from the source",
      "normalized_label": "lowercase normalized version",
      "evidence": "short verbatim snippet from table context",
      "sample_value": 1234.5
    }}
  ]
}}

sample_value is REQUIRED — it must be one actual numeric value for this field taken directly
from the table. If you cannot find a real number for a row, do not include that row at all.

{_SHARED_RULES}
""".strip()


_CASH_FLOW_SYSTEM_PROMPT = f"""
You are a financial statement row-label extractor specializing in CASH FLOW STATEMENTS
(also called Statements of Cash Flows).

You will receive text from PDF pages that have been pre-classified as cash flow statement pages.
Your job is to extract EVERY SINGLE row label that appears in the cash flow table(s).
Be exhaustive — extract all rows including sub-items within each section.

Also detect the reporting currency and unit (e.g. "EUR thousands", "USD millions").

Typical rows you should find (this list is NOT exhaustive — extract everything in the table):
Operating activities (extract each adjustment line separately):
  Net income / Profit for the year (starting point), Income before income taxes,
  Depreciation and amortization, Impairment losses, Share-based compensation,
  Unrealized foreign exchange gains/losses, Changes in trade receivables,
  Changes in inventories, Changes in trade payables, Changes in other working capital,
  Changes in contract liabilities, Income taxes paid, Interest paid, Interest received,
  Dividends received, Other operating cash flows,
  Net cash from operating activities, Cash generated from operations
Investing activities (extract each line separately):
  Capital expenditure, Purchases of property plant and equipment,
  Purchases of intangible assets, Proceeds from disposal of PP&E,
  Acquisitions net of cash, Proceeds from divestitures,
  Purchases of investments / financial assets, Proceeds from sale of investments,
  Loans granted, Repayment of loans, Other investing cash flows,
  Net cash from investing activities, Net cash used in investing activities
Financing activities (extract each line separately):
  Proceeds from borrowings, Repayment of borrowings, Net change in borrowings,
  Payment of lease liabilities, Dividends paid, Share buybacks / repurchases,
  Proceeds from share issuance, Proceeds from exercise of options,
  Other financing cash flows,
  Net cash from financing activities, Net cash used in financing activities
Reconciliation:
  Effect of exchange rate changes on cash, Net change in cash,
  Cash at beginning of period, Cash at end of period,
  Opening cash and cash equivalents, Closing cash and cash equivalents,
  Free cash flow (if shown as a presented line)

Return JSON only with this shape:
{{
  "currency": "e.g. EUR thousands, USD millions, GBP thousands — or null if not found",
  "fields": [
    {{
      "raw_label": "exact label text from the source",
      "normalized_label": "lowercase normalized version",
      "evidence": "short verbatim snippet from table context",
      "sample_value": 1234.5
    }}
  ]
}}

sample_value is REQUIRED — it must be one actual numeric value for this field taken directly
from the table. If you cannot find a real number for a row, do not include that row at all.

{_SHARED_RULES}
""".strip()


_STATEMENT_SYSTEM_PROMPTS: dict[str, str] = {
    "income_statement": _INCOME_SYSTEM_PROMPT,
    "balance_sheet": _BALANCE_SHEET_SYSTEM_PROMPT,
    "cash_flow_statement": _CASH_FLOW_SYSTEM_PROMPT,
}

_STATEMENT_LABEL: dict[str, str] = {
    "income_statement": "Income Statement",
    "balance_sheet": "Balance Sheet",
    "cash_flow_statement": "Cash Flow Statement",
}

# Max characters per statement to send to the LLM (3 calls × ~50K each is reasonable)
_MAX_CHARS_PER_STATEMENT = 60_000

# Footnote refs like "(A.1)", "(C.2)", "B.6" — same pattern as table_cleaner
_FOOTNOTE_REF = re.compile(
    r"^[\s(]*[A-Za-z][\d]+(?:[.,]\d+)*[\s)]*"
    r"(?:[,;]\s*[\s(]*[A-Za-z][\d]+(?:[.,]\d+)*[\s)]*)*$"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredField:
    raw_label: str
    normalized_label: str
    evidence: str
    statement_type: str


@dataclass
class DiscoveryResult:
    company: str
    report_year: int | None
    fields: list[DiscoveredField] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    currency: str | None = None  # e.g. "EUR thousands", "USD millions"

    def by_statement(self) -> dict[str, list[dict[str, str]]]:
        grouped = {
            "income_statement": [],
            "balance_sheet": [],
            "cash_flow_statement": [],
        }
        for f in self.fields:
            grouped[f.statement_type].append(
                {
                    "raw_label": f.raw_label,
                    "normalized_label": f.normalized_label,
                    "evidence": f.evidence,
                }
            )
        return grouped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_label(label: str) -> str:
    return " ".join((label or "").strip().lower().replace("&", "and").split())


def _dedupe_key(statement_type: str, raw_label: str, normalized_label: str) -> tuple[str, str]:
    preferred = normalized_label or _normalize_label(raw_label)
    return statement_type, preferred


def _looks_like_bad_label(label: str) -> bool:
    s = " ".join((label or "").split()).strip()
    if not s:
        return True

    if len(s) <= 1:
        return True

    # Purely numeric/symbolic — not a label
    if re.fullmatch(r"[\d\s,.()\/%$€£¥\-–—]+", s):
        return True

    # Footnote reference like "Note 3", "note 1a"
    if re.fullmatch(r"(note|notes?)\s*\d+[a-z]?", s, re.IGNORECASE):
        return True

    # Section numbering artifact like "A.1", "C.2.3"
    if re.fullmatch(r"[A-Z]?\d+(?:\.\d+)*", s):
        return True

    # Footnote ref like "(A.1)", "(C.2)" — mirrors table_cleaner._FOOTNOTE_REF
    if _FOOTNOTE_REF.match(s):
        return True

    # Too few actual alpha characters — truncated cell fragment
    if sum(1 for c in s if c.isalpha()) < 4:
        return True

    return False


def _build_user_prompt(
    company: str,
    report_year: int | None,
    statement_type: str,
    page_texts: list[str],
) -> str:
    year_str = str(report_year) if report_year is not None else "unknown"
    combined = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
    # Truncate to keep LLM call manageable
    if len(combined) > _MAX_CHARS_PER_STATEMENT:
        combined = combined[:_MAX_CHARS_PER_STATEMENT]

    stmt_label = _STATEMENT_LABEL.get(statement_type, statement_type)

    return f"""
Company: {company}
Report year: {year_str}
Statement: {stmt_label}

Task:
Extract every row label from the {stmt_label} table(s) in the text below.
These pages were detected as {stmt_label} pages from the annual report PDF.
Focus on rows that have numeric values in adjacent columns.

Instructions:
- Only extract labels that appear as table rows with associated numeric values.
- Preserve industry-specific label wording exactly as it appears.
- Evidence must be a short snippet from the actual table text (not from narrative paragraphs).
- Ignore narrative text, commentary, and prose paragraphs.

Page text:
{combined}
""".strip()


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def discover_company_fields(
    *,
    company: str,
    report_year: int | None,
    pages_by_statement: dict[str, list[str]],
    client: LLMClient | None = None,
) -> DiscoveryResult:
    """
    Run one focused LLM call per statement type on pre-classified page texts.

    Args:
        company: Company slug.
        report_year: Fiscal year integer, or None.
        pages_by_statement: Dict mapping statement_type -> list of page text strings.
            Keys: "income_statement", "balance_sheet", "cash_flow_statement".
            Each value is a list of page texts already classified as that statement.
        client: LLMClient instance (created if not provided).

    Returns:
        DiscoveryResult with fields from all three statements combined.
    """
    client = client or LLMClient()

    def _call_one(statement_type: str) -> tuple[str, dict[str, Any]]:
        """One LLM call for one statement type. Returns (statement_type, raw data)."""
        page_texts = [p for p in (pages_by_statement.get(statement_type) or []) if p and p.strip()]
        if not page_texts:
            logger.info("[discovery] %s/%s — no pages, skipping", company, statement_type)
            return statement_type, {}

        system_prompt = _STATEMENT_SYSTEM_PROMPTS[statement_type]
        user_prompt = _build_user_prompt(company, report_year, statement_type, page_texts)
        logger.info(
            "[discovery] %s/%s — calling LLM with %d page(s), ~%d chars",
            company, statement_type, len(page_texts), len(user_prompt),
        )
        try:
            response = client.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            return statement_type, response.content if isinstance(response.content, dict) else {}
        except Exception as exc:
            logger.warning("[discovery] %s/%s — LLM error: %s", company, statement_type, exc)
            return statement_type, {}

    # Fire all three statement-type calls in parallel
    all_fields: list[DiscoveredField] = []
    raw_responses: dict[str, Any] = {}
    seen: set[tuple[str, str]] = set()
    detected_currency: str | None = None

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_call_one, stmt): stmt
            for stmt in ("income_statement", "balance_sheet", "cash_flow_statement")
        }
        # Process results in statement order for deterministic deduplication
        results_by_stmt: dict[str, dict] = {}
        for future in as_completed(futures):
            stmt_type, data = future.result()
            results_by_stmt[stmt_type] = data

    for statement_type in ("income_statement", "balance_sheet", "cash_flow_statement"):
        data = results_by_stmt.get(statement_type, {})
        items = data.get("fields", [])
        if not isinstance(items, list):
            items = []

        # Take the first non-null currency seen across the three responses
        if detected_currency is None:
            raw_currency = (data.get("currency") or "").strip()
            if raw_currency and raw_currency.lower() not in ("null", "none", "unknown"):
                detected_currency = raw_currency

        raw_responses[statement_type] = items
        statement_fields = 0

        for item in items:
            if not isinstance(item, dict):
                continue

            raw_label = " ".join((item.get("raw_label") or "").split()).strip()
            evidence = " ".join((item.get("evidence") or "").split()).strip()
            normalized_label = " ".join((item.get("normalized_label") or "").split()).strip()

            if not raw_label or not evidence:
                continue

            if _looks_like_bad_label(raw_label):
                continue

            # Require a confirmed numeric value — fields without one are hallucinations
            sample_value = item.get("sample_value")
            if not isinstance(sample_value, (int, float)):
                continue

            if not normalized_label:
                normalized_label = _normalize_label(raw_label)
            else:
                normalized_label = _normalize_label(normalized_label)

            key = _dedupe_key(statement_type, raw_label, normalized_label)
            if key in seen:
                continue
            seen.add(key)

            all_fields.append(
                DiscoveredField(
                    raw_label=raw_label,
                    normalized_label=normalized_label,
                    evidence=evidence,
                    statement_type=statement_type,
                )
            )
            statement_fields += 1

        logger.info(
            "[discovery] %s/%s — kept %d fields (sample_value validated)",
            company, statement_type, statement_fields,
        )

    return DiscoveryResult(
        company=company,
        report_year=report_year,
        fields=all_fields,
        raw_response=raw_responses,
        currency=detected_currency,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_discovery_result(
    result: DiscoveryResult,
    *,
    output_dir: str | Path = "data/company_field_catalogs",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_part = str(result.report_year) if result.report_year is not None else "unknown"
    path = out_dir / f"{result.company}_{year_part}.json"

    payload = {
        "company": result.company,
        "report_year": result.report_year,
        "currency": result.currency,
        "fields": result.by_statement(),
        "raw_response": result.raw_response,
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved field catalog: %s", path)
    return path


def load_discovery_result(
    *,
    company: str,
    report_year: int | None,
    input_dir: str | Path = "data/company_field_catalogs",
) -> dict[str, Any]:
    year_part = str(report_year) if report_year is not None else "unknown"
    path = Path(input_dir) / f"{company}_{year_part}.json"
    return json.loads(path.read_text(encoding="utf-8"))
