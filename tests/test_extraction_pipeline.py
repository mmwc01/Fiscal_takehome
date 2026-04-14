"""
Tests for fiscal/extraction/pipeline.py

All pdfplumber I/O is mocked — hand-crafted PDFs cannot produce tables via
pdfplumber's layout engine, so we mock pdfplumber.open() to return controlled
page objects with preset text and table data.

Test groups
-----------
  TestRunExtract          — orchestration logic, filing selection, return counts
  TestExtractOne          — per-filing dispatch: missing PDF, DB status updates
  TestExtractPdf          — core page loop: trigger detection, table storage
  TestReExtraction        — --filing-id path: old raw_tables cleared then re-stored
  TestExtractFilingPublic — public extract_filing() wrapper (never raises)
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fiscal import db
from fiscal.extraction import pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(text: str, tables: list) -> MagicMock:
    """Return a mock pdfplumber page with preset text and tables."""
    page = MagicMock()
    page.page_number = 1
    page.extract_text.return_value = text
    page.extract_tables.return_value = tables
    return page


def _make_pdf_mock(pages: list[MagicMock]) -> MagicMock:
    """Return a context-manager-compatible pdfplumber PDF mock."""
    pdf = MagicMock()
    pdf.pages = pages
    pdf.__enter__ = lambda s: s
    pdf.__exit__ = MagicMock(return_value=False)
    return pdf


def _insert_filing(tmp_db: Path, *, status: str = "downloaded", local_path: str | None = None) -> int:
    """Insert a minimal filing row and return its id."""
    with db.get_connection(tmp_db) as conn:
        conn.execute(
            """
            INSERT INTO filings
                (company, title, url, document_type, fiscal_year,
                 status, discovered_at, metadata, local_path)
            VALUES
                ('adyen','Annual Report 2023',
                 'https://adyen.com/ar2023.pdf','annual_report',2023,
                 ?, datetime('now'), '{}', ?)
            """,
            (status, local_path),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _get_filing(tmp_db: Path, filing_id: int) -> dict:
    with db.get_connection(tmp_db) as conn:
        return db.get_filing_by_id(conn, filing_id)


def _get_raw_tables(tmp_db: Path, filing_id: int) -> list[dict]:
    with db.get_connection(tmp_db) as conn:
        return db.get_raw_tables(conn, filing_id)


# ---------------------------------------------------------------------------
# TestRunExtract
# ---------------------------------------------------------------------------

class TestRunExtract:
    def test_no_filings_returns_zero_counts(self, tmp_db):
        counts = pipeline.run_extract(db_path=tmp_db)
        assert counts == {"extracted": 0, "failed": 0, "skipped": 0}

    def test_unknown_filing_id_raises(self, tmp_db):
        with pytest.raises(ValueError, match="not found"):
            pipeline.run_extract(filing_id=999, db_path=tmp_db)

    def test_only_downloaded_annual_reports_selected(self, tmp_db, tmp_path):
        """Filings with status != 'downloaded' or type != 'annual_report' are ignored."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with db.get_connection(tmp_db) as conn:
            # This one should be selected
            conn.execute(
                "INSERT INTO filings (company,title,url,document_type,fiscal_year,"
                "status,discovered_at,metadata,local_path) VALUES "
                "('adyen','AR','https://a.com/1.pdf','annual_report',2023,"
                "'downloaded',datetime('now'),'{}',?)",
                (str(pdf),),
            )
            # Wrong status — should be skipped
            conn.execute(
                "INSERT INTO filings (company,title,url,document_type,fiscal_year,"
                "status,discovered_at,metadata,local_path) VALUES "
                "('adyen','AR','https://a.com/2.pdf','annual_report',2023,"
                "'discovered',datetime('now'),'{}',?)",
                (str(pdf),),
            )
            # Wrong doc type — should be skipped
            conn.execute(
                "INSERT INTO filings (company,title,url,document_type,fiscal_year,"
                "status,discovered_at,metadata,local_path) VALUES "
                "('adyen','Interim','https://a.com/3.pdf','interim_report',2023,"
                "'downloaded',datetime('now'),'{}',?)",
                (str(pdf),),
            )

        page = _make_page("Consolidated Income Statement\nRevenue 1000 900",
                          [[["Revenue", "1000", "900"]]])
        page.page_number = 1

        pdf_mock = _make_pdf_mock([page])

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = pdf_mock
            counts = pipeline.run_extract(db_path=tmp_db)

        assert counts["extracted"] == 1
        assert counts["skipped"] == 0

    def test_company_filter_applied(self, tmp_db, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with db.get_connection(tmp_db) as conn:
            for company, url in [("adyen", "https://a.com/1.pdf"),
                                  ("heineken", "https://b.com/1.pdf")]:
                conn.execute(
                    "INSERT INTO filings (company,title,url,document_type,fiscal_year,"
                    "status,discovered_at,metadata,local_path) VALUES "
                    f"('{company}','AR','{url}','annual_report',2023,"
                    "'downloaded',datetime('now'),'{}',?)",
                    (str(pdf),),
                )

        page = _make_page("Balance Sheet\nTotal assets 1000 900",
                          [[["Total assets", "1000", "900"]]])
        page.page_number = 1

        pdf_mock = _make_pdf_mock([page])

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = pdf_mock
            counts = pipeline.run_extract(company="adyen", db_path=tmp_db)

        assert counts["extracted"] == 1

    def test_extracted_count_incremented_on_success(self, tmp_db, tmp_path):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        page = _make_page("Consolidated Balance Sheet\nTotal assets 65000 61000",
                          [[["Total assets", "65000", "61000"]]])
        page.page_number = 1

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = _make_pdf_mock([page])
            counts = pipeline.run_extract(db_path=tmp_db)

        assert counts["extracted"] == 1
        assert counts["failed"] == 0
        assert counts["skipped"] == 0

    def test_failed_count_incremented_on_missing_pdf(self, tmp_db):
        _insert_filing(tmp_db, status="downloaded", local_path="/nonexistent/ar.pdf")
        counts = pipeline.run_extract(db_path=tmp_db)
        assert counts["failed"] == 1

    def test_failed_count_incremented_on_pdfplumber_error(self, tmp_db, tmp_path):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.side_effect = Exception("corrupt PDF")
            counts = pipeline.run_extract(db_path=tmp_db)

        assert counts["failed"] == 1
        assert counts["extracted"] == 0


# ---------------------------------------------------------------------------
# TestExtractOne
# ---------------------------------------------------------------------------

class TestExtractOne:
    def test_missing_local_path_marks_skipped_not_exception(self, tmp_db):
        """A filing with no local_path should not raise — returns skipped."""
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=None)
        with db.get_connection(tmp_db) as conn:
            filing = db.get_filing_by_id(conn, filing_id)
        result = pipeline._extract_one(filing, tmp_db)
        assert result.get("skipped")

    def test_nonexistent_file_marks_extraction_failed_in_db(self, tmp_db):
        filing_id = _insert_filing(
            tmp_db, status="downloaded", local_path="/no/such/file.pdf"
        )
        with db.get_connection(tmp_db) as conn:
            filing = db.get_filing_by_id(conn, filing_id)
        pipeline._extract_one(filing, tmp_db)

        row = _get_filing(tmp_db, filing_id)
        assert row["status"] == "extraction_failed"
        assert row["extraction_error"] is not None

    def test_successful_extraction_marks_status_extracted(self, tmp_db, tmp_path):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        page = _make_page("Cash Flow Statement\nOperating activities 500 400",
                          [[["Operating activities", "500", "400"]]])
        page.page_number = 1

        with db.get_connection(tmp_db) as conn:
            filing = db.get_filing_by_id(conn, filing_id)

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = _make_pdf_mock([page])
            pipeline._extract_one(filing, tmp_db)

        row = _get_filing(tmp_db, filing_id)
        assert row["status"] == "extracted"
        assert row["extracted_at"] is not None

    def test_pdfplumber_exception_marks_extraction_failed(self, tmp_db, tmp_path):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        with db.get_connection(tmp_db) as conn:
            filing = db.get_filing_by_id(conn, filing_id)

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.side_effect = RuntimeError("bad PDF")
            pipeline._extract_one(filing, tmp_db)

        row = _get_filing(tmp_db, filing_id)
        assert row["status"] == "extraction_failed"
        assert "bad PDF" in row["extraction_error"]


# ---------------------------------------------------------------------------
# TestExtractPdf — core page-scanning logic
# ---------------------------------------------------------------------------

class TestExtractPdf:
    def _run(self, tmp_db, tmp_path, pages):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))
        with db.get_connection(tmp_db) as conn:
            filing = db.get_filing_by_id(conn, filing_id)
        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = _make_pdf_mock(pages)
            result = pipeline._extract_pdf(filing, tmp_db)
        return filing_id, result

    def test_triggered_page_stores_raw_table(self, tmp_db, tmp_path):
        table_data = [["Revenue", "1,234", "1,100"], ["Net income", "234", "200"]]
        page = _make_page(
            "Consolidated Income Statement\nRevenue 1234 1100\nNet income 234 200",
            [table_data],
        )
        page.page_number = 1

        filing_id, result = self._run(tmp_db, tmp_path, [page])

        assert result["tables_stored"] == 1
        assert result["pages_triggered"] == 1

        rows = _get_raw_tables(tmp_db, filing_id)
        assert len(rows) == 1
        assert rows[0]["statement_type"] == "income_statement"
        assert rows[0]["page_number"] == 1
        assert rows[0]["table_index"] == 0
        assert rows[0]["num_rows"] == 2
        assert rows[0]["num_cols"] == 3
        assert json.loads(rows[0]["raw_data"]) == table_data

    def test_non_triggered_page_stores_nothing(self, tmp_db, tmp_path):
        page = _make_page("Notes to the Financial Statements\nSee Note 3.", [])
        page.page_number = 2

        filing_id, result = self._run(tmp_db, tmp_path, [page])

        assert result["tables_stored"] == 0
        assert result["pages_triggered"] == 0
        assert _get_raw_tables(tmp_db, filing_id) == []

    def test_empty_page_text_skipped(self, tmp_db, tmp_path):
        page = _make_page("", [])
        page.page_number = 1
        page.extract_text.return_value = None  # pdfplumber can return None

        filing_id, result = self._run(tmp_db, tmp_path, [page])

        assert result["pages_triggered"] == 0
        assert result["tables_stored"] == 0

    def test_multiple_tables_on_one_page(self, tmp_db, tmp_path):
        tables = [
            [["Revenue", "1000"], ["EBITDA", "500"]],
            [["Net income", "200"], ["EPS", "1.5"]],
        ]
        page = _make_page(
            "Consolidated Income Statement\nRevenue 1000\nEBITDA 500\nNet income 200",
            tables,
        )
        page.page_number = 1

        filing_id, result = self._run(tmp_db, tmp_path, [page])

        assert result["tables_stored"] == 2
        rows = _get_raw_tables(tmp_db, filing_id)
        assert len(rows) == 2
        assert rows[0]["table_index"] == 0
        assert rows[1]["table_index"] == 1

    def test_multiple_pages_mixed_triggers(self, tmp_db, tmp_path):
        page1 = _make_page("Corporate governance and overview", [])
        page1.page_number = 1

        page2 = _make_page(
            "Consolidated Balance Sheet\nTotal assets 65000",
            [[["Total assets", "65000", "61000"]]],
        )
        page2.page_number = 2

        page3 = _make_page(
            "Statement of Cash Flows\nOperating activities 1000",
            [[["Operating activities", "1000", "900"]]],
        )
        page3.page_number = 3

        filing_id, result = self._run(tmp_db, tmp_path, [page1, page2, page3])

        assert result["pages_scanned"] == 3
        assert result["pages_triggered"] == 2
        assert result["tables_stored"] == 2

        rows = _get_raw_tables(tmp_db, filing_id)
        types = {r["statement_type"] for r in rows}
        assert types == {"balance_sheet", "cash_flow_statement"}

    def test_raw_data_json_roundtrip(self, tmp_db, tmp_path):
        table_data = [["Label", "2023", "2022"], ["Revenue", "1,234", None], ["EBITDA", "567", "489"]]
        page = _make_page(
            "Income Statement\nRevenue 1234\nEBITDA 567",
            [table_data],
        )
        page.page_number = 1

        filing_id, _ = self._run(tmp_db, tmp_path, [page])

        rows = _get_raw_tables(tmp_db, filing_id)
        assert json.loads(rows[0]["raw_data"]) == table_data

    def test_detection_reason_stored(self, tmp_db, tmp_path):
        page = _make_page("Consolidated Balance Sheet\nTotal assets 65000", [[["Total assets", "65000"]]])
        page.page_number = 1

        filing_id, _ = self._run(tmp_db, tmp_path, [page])
        rows = _get_raw_tables(tmp_db, filing_id)
        assert rows[0]["detection_reason"]
        assert "balance sheet" in rows[0]["detection_reason"].lower()

    def test_pages_scanned_counts_all_pages(self, tmp_db, tmp_path):
        pages = []
        for i in range(1, 6):
            p = _make_page("Some text about operations", [])
            p.page_number = i
            pages.append(p)

        _, result = self._run(tmp_db, tmp_path, pages)
        assert result["pages_scanned"] == 5
        assert result["pages_triggered"] == 0

    def test_empty_table_skipped(self, tmp_db, tmp_path):
        """pdfplumber can return empty lists inside extract_tables — skip them."""
        page = _make_page(
            "Consolidated Income Statement\nRevenue 1000",
            [[], [["Revenue", "1000"]]],  # first table is empty
        )
        page.page_number = 1

        filing_id, result = self._run(tmp_db, tmp_path, [page])
        assert result["tables_stored"] == 1


# ---------------------------------------------------------------------------
# TestReExtraction — --filing-id clears old data then re-extracts
# ---------------------------------------------------------------------------

class TestReExtraction:
    def test_filing_id_path_clears_existing_raw_tables(self, tmp_db, tmp_path):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        # Seed some existing raw_table rows
        with db.get_connection(tmp_db) as conn:
            for i in range(3):
                db.insert_raw_table(conn, {
                    "filing_id": filing_id,
                    "page_number": i + 1,
                    "table_index": 0,
                    "raw_data": "[]",
                    "num_rows": 0,
                    "num_cols": 0,
                    "statement_type": "income_statement",
                    "detection_reason": "old run",
                    "extracted_at": "2024-01-01T00:00:00+00:00",
                })

        assert len(_get_raw_tables(tmp_db, filing_id)) == 3

        page = _make_page(
            "Consolidated Cash Flow Statement\nOperating activities 800",
            [[["Operating activities", "800", "700"]]],
        )
        page.page_number = 1

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = _make_pdf_mock([page])
            pipeline.run_extract(filing_id=filing_id, db_path=tmp_db)

        rows = _get_raw_tables(tmp_db, filing_id)
        # Only the new table from re-extraction should remain
        assert len(rows) == 1
        assert rows[0]["statement_type"] == "cash_flow_statement"

    def test_filing_id_path_returns_extracted_count_1(self, tmp_db, tmp_path):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        page = _make_page(
            "Consolidated Balance Sheet\nTotal assets 65000 61000",
            [[["Total assets", "65000", "61000"]]],
        )
        page.page_number = 1

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = _make_pdf_mock([page])
            counts = pipeline.run_extract(filing_id=filing_id, db_path=tmp_db)

        assert counts["extracted"] == 1
        assert counts["failed"] == 0

    def test_filing_id_with_no_tables_still_marks_extracted(self, tmp_db, tmp_path):
        """A filing where no pages trigger should still be marked 'extracted'."""
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        page = _make_page("About our company. We made progress this year.", [])
        page.page_number = 1

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = _make_pdf_mock([page])
            pipeline.run_extract(filing_id=filing_id, db_path=tmp_db)

        row = _get_filing(tmp_db, filing_id)
        assert row["status"] == "extracted"


# ---------------------------------------------------------------------------
# TestExtractFilingPublic — public API wrapper never raises
# ---------------------------------------------------------------------------

class TestExtractFilingPublic:
    def test_nonexistent_filing_returns_error_dict(self, tmp_db):
        result = pipeline.extract_filing(filing_id=999, db_path=tmp_db)
        assert "error" in result
        assert "999" in result["error"]

    def test_successful_extraction_returns_stats_dict(self, tmp_db, tmp_path):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        page = _make_page(
            "Consolidated Income Statement\nRevenue 1234\nNet income 234",
            [[["Revenue", "1234"], ["Net income", "234"]]],
        )
        page.page_number = 1

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = _make_pdf_mock([page])
            result = pipeline.extract_filing(filing_id=filing_id, db_path=tmp_db)

        assert "error" not in result
        assert result["tables_stored"] == 1
        assert result["pages_triggered"] == 1
        assert result["pages_scanned"] == 1

    def test_pdf_error_returns_error_dict_not_exception(self, tmp_db, tmp_path):
        pdf_file = tmp_path / "ar.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        filing_id = _insert_filing(tmp_db, status="downloaded", local_path=str(pdf_file))

        with patch("fiscal.extraction.pipeline.pdfplumber") as mock_plumber:
            mock_plumber.open.side_effect = Exception("unreadable")
            result = pipeline.extract_filing(filing_id=filing_id, db_path=tmp_db)

        assert "error" in result
        assert "unreadable" in result["error"]
