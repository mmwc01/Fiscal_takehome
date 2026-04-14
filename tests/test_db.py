"""Tests for fiscal.db — schema initialization and query helpers."""
from fiscal import db


class TestInitDb:
    def test_creates_all_tables(self, tmp_db):
        with db.get_connection(tmp_db) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        expected = {"filings", "raw_tables", "raw_line_items", "compiled_views"}
        assert expected.issubset(tables)

    def test_idempotent(self, tmp_db):
        """Calling init_db twice must not raise."""
        db.init_db(tmp_db)
        db.init_db(tmp_db)

    def test_wal_mode(self, tmp_db):
        with db.get_connection(tmp_db) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


class TestUpsertFiling:
    def _sample(self, url="https://example.com/ar-2023.pdf"):
        return {
            "company":       "adyen",
            "title":         "Annual Report 2023",
            "url":           url,
            "document_type": "annual_report",
            "fiscal_year":   2023,
            "status":        "classified",
        }

    def test_insert_new_returns_true(self, conn):
        assert db.upsert_filing(conn, self._sample()) is True

    def test_duplicate_url_returns_false(self, conn):
        data = self._sample()
        db.upsert_filing(conn, data)
        assert db.upsert_filing(conn, data) is False

    def test_row_count_stays_one_on_duplicate(self, conn):
        data = self._sample()
        db.upsert_filing(conn, data)
        db.upsert_filing(conn, data)
        rows = db.get_filings(conn)
        assert len(rows) == 1

    def test_different_urls_both_inserted(self, conn):
        db.upsert_filing(conn, self._sample("https://example.com/a.pdf"))
        db.upsert_filing(conn, self._sample("https://example.com/b.pdf"))
        assert len(db.get_filings(conn)) == 2


class TestGetFilings:
    def _insert(self, conn, company, url, doc_type="annual_report", status="classified"):
        db.upsert_filing(conn, {
            "company": company,
            "url": url,
            "document_type": doc_type,
            "status": status,
        })

    def test_filter_by_company(self, conn):
        self._insert(conn, "adyen",    "https://example.com/a.pdf")
        self._insert(conn, "heineken", "https://example.com/b.pdf")
        rows = db.get_filings(conn, company="adyen")
        assert len(rows) == 1
        assert rows[0]["company"] == "adyen"

    def test_filter_by_document_type(self, conn):
        self._insert(conn, "adyen", "https://example.com/ar.pdf", doc_type="annual_report")
        self._insert(conn, "adyen", "https://example.com/sr.pdf", doc_type="sustainability")
        rows = db.get_filings(conn, document_type="annual_report")
        assert len(rows) == 1

    def test_filter_by_status(self, conn):
        self._insert(conn, "adyen", "https://example.com/a.pdf", status="classified")
        self._insert(conn, "adyen", "https://example.com/b.pdf", status="downloaded")
        rows = db.get_filings(conn, status="downloaded")
        assert len(rows) == 1

    def test_no_filter_returns_all(self, conn):
        self._insert(conn, "adyen",    "https://example.com/a.pdf")
        self._insert(conn, "heineken", "https://example.com/b.pdf")
        assert len(db.get_filings(conn)) == 2

    def test_returns_dicts(self, conn):
        self._insert(conn, "adyen", "https://example.com/a.pdf")
        rows = db.get_filings(conn)
        assert isinstance(rows[0], dict)
        assert "company" in rows[0]


class TestGetStatusSummary:
    def test_returns_one_row_per_company(self, conn):
        for url in ("https://a.com/1.pdf", "https://a.com/2.pdf"):
            db.upsert_filing(conn, {
                "company": "adyen", "url": url,
                "document_type": "annual_report", "status": "classified",
            })
        db.upsert_filing(conn, {
            "company": "heineken", "url": "https://b.com/1.pdf",
            "document_type": "annual_report", "status": "classified",
        })
        rows = db.get_status_summary(conn)
        companies = [r["company"] for r in rows]
        assert "adyen" in companies
        assert "heineken" in companies

    def test_annual_report_count(self, conn):
        db.upsert_filing(conn, {
            "company": "adyen", "url": "https://a.com/ar.pdf",
            "document_type": "annual_report", "status": "classified",
        })
        db.upsert_filing(conn, {
            "company": "adyen", "url": "https://a.com/other.pdf",
            "document_type": "other", "status": "classified",
        })
        rows = db.get_status_summary(conn)
        adyen_row = next(r for r in rows if r["company"] == "adyen")
        assert adyen_row["annual_reports"] == 1
        assert adyen_row["total_filings"] == 2
