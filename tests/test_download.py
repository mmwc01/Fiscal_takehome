"""
Tests for fiscal/download.py.

All network calls are mocked with the `responses` library.
File I/O uses pytest's tmp_path fixture via the tmp_db fixture (which places
the DB — and therefore the pdfs/ dir — inside a temp directory).
"""
import hashlib
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import responses as rsps

from fiscal import db
from fiscal import download


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_PDF = b"%PDF-1.4 fake content for testing"
_FAKE_SHA256 = hashlib.sha256(_FAKE_PDF).hexdigest()
_ADYEN_URL = "https://media.adyen.com/documents/adyen-annual-report-2023.pdf"


@pytest.fixture
def db_with_annual_report(tmp_db):
    """Insert one classified annual_report filing for adyen."""
    with db.get_connection(tmp_db) as conn:
        db.upsert_filing(conn, {
            "company":       "adyen",
            "title":         "Annual Report 2023",
            "url":           _ADYEN_URL,
            "document_type": "annual_report",
            "fiscal_year":   2023,
            "status":        "classified",
        })
    return tmp_db


@pytest.fixture
def db_with_non_annual(tmp_db):
    """Insert a sustainability filing — should never be downloaded."""
    with db.get_connection(tmp_db) as conn:
        db.upsert_filing(conn, {
            "company":       "adyen",
            "title":         "Sustainability Report 2023",
            "url":           "https://example.com/sustainability-2023.pdf",
            "document_type": "sustainability",
            "fiscal_year":   2023,
            "status":        "classified",
        })
    return tmp_db


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------

class TestSafeFilename:
    def test_uses_url_basename(self):
        filing = {"url": _ADYEN_URL, "company": "adyen", "fiscal_year": 2023}
        name = download._safe_filename(filing)
        assert name == "adyen-annual-report-2023.pdf"

    def test_falls_back_to_title_when_url_basename_too_short(self):
        filing = {
            "url": "https://example.com/ar.pdf",  # basename is just "ar.pdf" — too short
            "title": "Annual Report 2022",
            "company": "adyen",
            "fiscal_year": 2022,
        }
        name = download._safe_filename(filing)
        assert name.endswith(".pdf")
        assert "annual" in name.lower()

    def test_sanitizes_unsafe_characters(self):
        filing = {
            "url": "https://example.com/report (2023) [final].pdf",
            "company": "adyen",
            "fiscal_year": 2023,
        }
        name = download._safe_filename(filing)
        assert not re.search(r"[^\w.\-]", name), f"Unsafe chars in: {name}"

    def test_enforces_max_length(self):
        filing = {
            "url": "https://example.com/" + ("x" * 300) + ".pdf",
            "company": "adyen",
            "fiscal_year": 2023,
        }
        name = download._safe_filename(filing)
        assert len(name) <= 200


# ---------------------------------------------------------------------------
# _compute_sha256
# ---------------------------------------------------------------------------

class TestComputeSha256:
    def test_matches_known_hash(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(_FAKE_PDF)
        assert download._compute_sha256(f) == _FAKE_SHA256


# ---------------------------------------------------------------------------
# _dest_path
# ---------------------------------------------------------------------------

class TestDestPath:
    def test_path_structure(self, tmp_path):
        filing = {
            "url": _ADYEN_URL,
            "company": "adyen",
            "fiscal_year": 2023,
        }
        pdfs_root = tmp_path / "pdfs"
        dest = download._dest_path(filing, pdfs_root)
        assert dest.parent == pdfs_root / "adyen"
        assert dest.name == "adyen-annual-report-2023.pdf"


# ---------------------------------------------------------------------------
# run_download — successful download
# ---------------------------------------------------------------------------

class TestRunDownload:
    @rsps.activate
    def test_downloads_file_and_updates_db(self, db_with_annual_report):
        rsps.add(rsps.GET, _ADYEN_URL, body=_FAKE_PDF, status=200)

        counts = download.run_download(
            db_path=db_with_annual_report, rate_limit_secs=0
        )

        assert counts["downloaded"] == 1
        assert counts["skipped"] == 0
        assert counts["failed"] == 0

    @rsps.activate
    def test_db_updated_after_download(self, db_with_annual_report):
        rsps.add(rsps.GET, _ADYEN_URL, body=_FAKE_PDF, status=200)
        download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        with db.get_connection(db_with_annual_report) as conn:
            filings = db.get_filings(conn, company="adyen")
        filing = filings[0]

        assert filing["status"] == "downloaded"
        assert filing["sha256"] == _FAKE_SHA256
        assert filing["file_size_bytes"] == len(_FAKE_PDF)
        assert filing["local_path"] is not None
        assert filing["downloaded_at"] is not None

    @rsps.activate
    def test_file_exists_on_disk_after_download(self, db_with_annual_report):
        rsps.add(rsps.GET, _ADYEN_URL, body=_FAKE_PDF, status=200)
        download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        with db.get_connection(db_with_annual_report) as conn:
            filing = db.get_filings(conn, company="adyen")[0]

        assert Path(filing["local_path"]).exists()
        assert Path(filing["local_path"]).read_bytes() == _FAKE_PDF

    @rsps.activate
    def test_file_placed_in_correct_directory(self, db_with_annual_report):
        rsps.add(rsps.GET, _ADYEN_URL, body=_FAKE_PDF, status=200)
        download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        pdfs_root = db_with_annual_report.parent / "pdfs"
        expected_dir = pdfs_root / "adyen"
        assert expected_dir.is_dir()
        pdf_files = list(expected_dir.glob("*.pdf"))
        assert len(pdf_files) == 1


# ---------------------------------------------------------------------------
# run_download — skip logic
# ---------------------------------------------------------------------------

class TestSkipAlreadyDownloaded:
    @rsps.activate
    def test_skips_when_file_exists_and_checksum_matches(self, db_with_annual_report):
        # First download
        rsps.add(rsps.GET, _ADYEN_URL, body=_FAKE_PDF, status=200)
        download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        # Second run — file exists, checksum matches
        rsps.add(rsps.GET, _ADYEN_URL, body=_FAKE_PDF, status=200)
        counts = download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        assert counts["skipped"] == 1
        assert counts["downloaded"] == 0
        # Only one GET was made (the first run); second run should not call GET
        assert len(rsps.calls) == 1

    @rsps.activate
    def test_redownloads_when_file_missing(self, db_with_annual_report):
        # First download
        rsps.add(rsps.GET, _ADYEN_URL, body=_FAKE_PDF, status=200)
        download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        # Delete the file
        with db.get_connection(db_with_annual_report) as conn:
            filing = db.get_filings(conn, company="adyen")[0]
        Path(filing["local_path"]).unlink()

        # Second run — file gone, should re-download
        rsps.add(rsps.GET, _ADYEN_URL, body=_FAKE_PDF, status=200)
        counts = download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        assert counts["downloaded"] == 1
        assert counts["skipped"] == 0


# ---------------------------------------------------------------------------
# run_download — error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @rsps.activate
    def test_http_404_does_not_crash_pipeline(self, db_with_annual_report):
        rsps.add(rsps.GET, _ADYEN_URL, status=404)

        counts = download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        assert counts["failed"] == 1
        assert counts["downloaded"] == 0

    @rsps.activate
    def test_connection_error_does_not_crash_pipeline(self, db_with_annual_report):
        import requests as req
        rsps.add(rsps.GET, _ADYEN_URL, body=req.ConnectionError("unreachable"))

        counts = download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        assert counts["failed"] == 1
        assert counts["downloaded"] == 0

    @rsps.activate
    def test_failed_filing_status_not_updated(self, db_with_annual_report):
        """A failed download must not change the filing's status in the DB."""
        rsps.add(rsps.GET, _ADYEN_URL, status=500)
        download.run_download(db_path=db_with_annual_report, rate_limit_secs=0)

        with db.get_connection(db_with_annual_report) as conn:
            filing = db.get_filings(conn, company="adyen")[0]

        assert filing["status"] == "classified"  # unchanged

    @rsps.activate
    def test_one_failure_does_not_block_others(self, tmp_db):
        """With two filings, a failure on the first should not prevent the second."""
        url_fail = "https://media.adyen.com/documents/adyen-annual-report-2022.pdf"
        url_ok   = "https://media.adyen.com/documents/adyen-annual-report-2023.pdf"

        with db.get_connection(tmp_db) as conn:
            db.upsert_filing(conn, {
                "company": "adyen", "url": url_fail,
                "document_type": "annual_report", "fiscal_year": 2022,
                "status": "classified",
            })
            db.upsert_filing(conn, {
                "company": "adyen", "url": url_ok,
                "document_type": "annual_report", "fiscal_year": 2023,
                "status": "classified",
            })

        rsps.add(rsps.GET, url_fail, status=404)
        rsps.add(rsps.GET, url_ok,   body=_FAKE_PDF, status=200)

        counts = download.run_download(db_path=tmp_db, rate_limit_secs=0)

        assert counts["downloaded"] == 1
        assert counts["failed"] == 1


# ---------------------------------------------------------------------------
# run_download — document type filter
# ---------------------------------------------------------------------------

class TestDocumentTypeFilter:
    @rsps.activate
    def test_non_annual_filings_are_ignored(self, db_with_non_annual):
        counts = download.run_download(db_path=db_with_non_annual, rate_limit_secs=0)

        assert counts["downloaded"] == 0
        assert counts["skipped"] == 0
        assert counts["failed"] == 0
        assert len(rsps.calls) == 0  # no HTTP request made

    @rsps.activate
    def test_company_filter_restricts_scope(self, tmp_db):
        """--company adyen should not touch heineken filings."""
        with db.get_connection(tmp_db) as conn:
            db.upsert_filing(conn, {
                "company": "heineken",
                "url": "https://example.com/heineken-ar-2023.pdf",
                "document_type": "annual_report",
                "fiscal_year": 2023,
                "status": "classified",
            })

        counts = download.run_download(
            company="adyen", db_path=tmp_db, rate_limit_secs=0
        )

        assert counts["downloaded"] == 0
        assert len(rsps.calls) == 0
