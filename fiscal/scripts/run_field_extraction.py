"""
Strategy-2 field extraction + report pipeline.

Steps:
  1. Load the field catalog JSON (from run_company_field_discovery_from_pdfs).
  2. For each PDF in data/pdfs/{company}/, make ONE LLM call to extract values
     for every discovered field (all three statement types in one shot).
  3. Merge values across PDFs (same-year report is authoritative).
  4. Save merged values to data/company_field_values/{company}.json.
  5. Build a PDF report to output/{company}_llm_report.pdf.
  6. Optionally write rows into the classified_metrics DB table.

Total LLM calls: 1 per PDF (e.g. 10 PDFs = 10 calls).

Usage:
    python -m fiscal.scripts.run_field_extraction --company adyen
    python -m fiscal.scripts.run_field_extraction --company adyen --no-pdf --write-db
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fiscal.llm.client import LLMClient
from fiscal.llm.field_value_extractor import run_field_extraction

logger = logging.getLogger(__name__)

_STATEMENT_TYPES = ("income_statement", "balance_sheet", "cash_flow_statement")


def _find_latest_catalog(company: str, catalog_dir: Path) -> Path:
    candidates = sorted(catalog_dir.glob(f"{company}_*.json"))
    if not candidates:
        raise RuntimeError(
            f"No field catalog found for '{company}' in {catalog_dir}.\n"
            f"Run field discovery first:\n"
            f"  python -m fiscal.scripts.run_company_field_discovery_from_pdfs --company {company}"
        )

    def _year(p: Path) -> int:
        m = re.search(r"(\d{4})", p.stem.replace(company, ""))
        return int(m.group(1)) if m else 0

    return max(candidates, key=_year)


def _write_to_db(values_path: Path, db_path: Path | None = None) -> int:
    """
    Write extracted values into classified_metrics as classification_method='llm_strategy2'.

    Uses synthetic raw_line_items (row_index=-2 sentinel to distinguish from
    llm_recovered rows at row_index=-1).

    Returns count of rows inserted.
    """
    from fiscal import db

    db_path = db_path or db.DB_PATH
    data = json.loads(values_path.read_text(encoding="utf-8"))
    company = data["company"]
    now = datetime.now(timezone.utc).isoformat()
    _SYNTHETIC_IDX = -2
    inserted = 0

    with db.get_connection(db_path) as conn:
        # Clean up previous strategy-2 rows for this company
        conn.execute(
            "DELETE FROM raw_line_items WHERE company = ? AND row_index = ?",
            (company, _SYNTHETIC_IDX),
        )
        conn.execute(
            "DELETE FROM classified_metrics WHERE company = ? AND classification_method = 'llm_strategy2'",
            (company,),
        )

        for stmt in _STATEMENT_TYPES:
            for row in data["statements"].get(stmt, []):
                norm = row["normalized_label"]
                raw_label = row["raw_label"]
                for year_str, value in row.get("values_by_year", {}).items():
                    try:
                        period_year = int(year_str)
                        value = float(value)
                    except (ValueError, TypeError):
                        continue

                    # Find the filing for this period year
                    filing_row = conn.execute(
                        "SELECT id FROM filings WHERE company = ? AND fiscal_year = ? LIMIT 1",
                        (company, period_year),
                    ).fetchone()
                    if not filing_row:
                        continue
                    filing_id = filing_row[0]

                    raw_table_row = conn.execute(
                        """SELECT id FROM raw_tables
                           WHERE filing_id = ? AND statement_type = ?
                           ORDER BY page_number LIMIT 1""",
                        (filing_id, stmt),
                    ).fetchone()
                    if not raw_table_row:
                        raw_table_row = conn.execute(
                            "SELECT id FROM raw_tables WHERE filing_id = ? ORDER BY page_number LIMIT 1",
                            (filing_id,),
                        ).fetchone()
                    if not raw_table_row:
                        continue
                    raw_table_id = raw_table_row[0]

                    try:
                        cursor = conn.execute(
                            """INSERT INTO raw_line_items
                               (raw_table_id, filing_id, company, report_year, statement_type,
                                raw_label, raw_value, parsed_value, period_label, period_year, row_index)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (raw_table_id, filing_id, company, period_year, stmt,
                             raw_label, str(value), value, str(period_year), period_year, _SYNTHETIC_IDX),
                        )
                        rli_id = cursor.lastrowid

                        conn.execute(
                            """INSERT OR IGNORE INTO classified_metrics
                               (raw_line_item_id, company, report_year, statement_type,
                                raw_label, raw_value, parsed_value, period_label, period_year,
                                metric_group, metric_key, display_label,
                                classification_confidence, classification_method,
                                classification_score, classified_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (rli_id, company, period_year, stmt,
                             raw_label, str(value), value, str(period_year), period_year,
                             "unclassified_financial", norm, raw_label,
                             "high", "llm_strategy2", 1.0, now),
                        )
                        inserted += 1
                    except Exception as exc:
                        logger.warning("[db] %s/%s/%s — insert failed: %s", company, period_year, norm, exc)

    logger.info("[db] %s — inserted %d rows into classified_metrics", company, inserted)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract financial values from all PDFs and build a report (Strategy 2)."
    )
    parser.add_argument("--company", required=True, help="Company slug, e.g. adyen")
    parser.add_argument(
        "--catalog",
        default=None,
        help="Path to field catalog JSON. Defaults to latest in data/company_field_catalogs/",
    )
    parser.add_argument(
        "--pdf-dir", default=None,
        help="Directory of PDFs. Defaults to data/pdfs/{company}/",
    )
    parser.add_argument(
        "--values-output", default=None,
        help="Output path for values JSON. Defaults to data/company_field_values/{company}.json",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for the PDF report. Default: output/",
    )
    parser.add_argument(
        "--max-pdfs", type=int, default=10,
        help="Max PDFs to process (newest first). Default: 10",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip building the PDF report.",
    )
    parser.add_argument(
        "--write-db", action="store_true",
        help="Also write extracted values into the classified_metrics DB table.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = LLMClient()
    if not client.enabled():
        raise SystemExit("OPENAI_API_KEY is not set. Check your .env file.")

    catalog_dir = Path("data/company_field_catalogs")
    catalog_path = Path(args.catalog) if args.catalog else _find_latest_catalog(args.company, catalog_dir)
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else Path("data/pdfs") / args.company
    values_output = (
        Path(args.values_output)
        if args.values_output
        else Path("data/company_field_values") / f"{args.company}.json"
    )

    logger.info("Company:       %s", args.company)
    logger.info("Catalog:       %s", catalog_path)
    logger.info("PDF dir:       %s", pdf_dir)
    logger.info("Values output: %s", values_output)

    # Step 2: Extract values from all PDFs (1 LLM call each)
    values_path = run_field_extraction(
        args.company,
        catalog_path=catalog_path,
        pdf_dir=pdf_dir,
        client=client,
        output_path=values_output,
        max_pdfs=args.max_pdfs,
    )
    print(f"Values saved:  {values_path}")

    # Step 3a: Build PDF report
    if not args.no_pdf:
        from fiscal.llm_pdf_builder import build_pdf_from_values

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = build_pdf_from_values(values_path, output_dir=out_dir)
        print(f"PDF report:    {pdf_path}")

    # Step 3b: Write to DB (optional)
    if args.write_db:
        count = _write_to_db(values_path)
        print(f"DB rows written: {count}")


if __name__ == "__main__":
    main()
