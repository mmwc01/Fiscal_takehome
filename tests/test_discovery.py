"""
Tests for the discovery pipeline and company adapter conventions.

Network calls are mocked throughout — these tests never hit real IR pages.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from fiscal import db
from fiscal.companies import REGISTRY, adyen, heineken, sap


# ---------------------------------------------------------------------------
# Registry and adapter contract
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_three_companies_registered(self):
        assert set(REGISTRY.keys()) == {"adyen", "heineken", "sap"}

    def test_each_module_has_required_attributes(self):
        required = ("KEY", "NAME", "IR_URL", "FALLBACK_FILINGS", "discover_filings")
        for key, module in REGISTRY.items():
            for attr in required:
                assert hasattr(module, attr), f"{key!r} missing attribute {attr!r}"

    def test_discover_filings_is_callable(self):
        for key, module in REGISTRY.items():
            assert callable(module.discover_filings), f"{key!r}.discover_filings not callable"

    def test_fallback_filings_are_non_empty(self):
        for key, module in REGISTRY.items():
            assert len(module.FALLBACK_FILINGS) >= 4, (
                f"{key!r} FALLBACK_FILINGS has fewer than 4 entries"
            )

    def test_fallback_filings_have_required_fields(self):
        for key, module in REGISTRY.items():
            for i, filing in enumerate(module.FALLBACK_FILINGS):
                assert "title" in filing,       f"{key}[{i}] missing title"
                assert "url" in filing,         f"{key}[{i}] missing url"
                assert "fiscal_year" in filing, f"{key}[{i}] missing fiscal_year"
                assert filing["url"].endswith(".pdf"), (
                    f"{key}[{i}] url does not end in .pdf"
                )
                assert isinstance(filing["fiscal_year"], int), (
                    f"{key}[{i}] fiscal_year is not an int"
                )

    def test_fallback_fiscal_years_are_plausible(self):
        for key, module in REGISTRY.items():
            years = [f["fiscal_year"] for f in module.FALLBACK_FILINGS]
            for year in years:
                assert 2010 <= year <= 2030, (
                    f"{key!r} has implausible fiscal_year {year}"
                )


class TestAdyenFallbackCoverage:
    def test_covers_ipo_year(self):
        years = {f["fiscal_year"] for f in adyen.FALLBACK_FILINGS}
        assert 2018 in years, "Adyen IPO'd in 2018; 2018 report should be in fallback"

    def test_covers_recent_year(self):
        years = {f["fiscal_year"] for f in adyen.FALLBACK_FILINGS}
        assert max(years) >= 2022


class TestHEINEKENFallbackCoverage:
    def test_covers_ten_years(self):
        assert len(heineken.FALLBACK_FILINGS) >= 10


# ---------------------------------------------------------------------------
# Adapter scraper → fallback behavior
# ---------------------------------------------------------------------------

def _mock_session_raises(exc):
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = exc
    return session


def _mock_session_empty_page():
    session = MagicMock(spec=requests.Session)
    resp = MagicMock()
    resp.text = "<html><body><p>No PDF links here.</p></body></html>"
    resp.raise_for_status = MagicMock()
    session.get.return_value = resp
    return session


class TestAdyenAdapter:
    def test_fallback_on_connection_error(self):
        session = _mock_session_raises(requests.ConnectionError("unreachable"))
        result = adyen.discover_filings(session)
        assert result == adyen.FALLBACK_FILINGS

    def test_fallback_on_http_error(self):
        session = _mock_session_raises(requests.HTTPError("503"))
        result = adyen.discover_filings(session)
        assert result == adyen.FALLBACK_FILINGS

    def test_fallback_when_no_pdf_links_found(self):
        result = adyen.discover_filings(_mock_session_empty_page())
        assert result == adyen.FALLBACK_FILINGS

    def test_fallback_returns_list_of_dicts(self):
        result = adyen.discover_filings(_mock_session_raises(Exception("any")))
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)


class TestSapAdapter:
    def test_always_falls_back_on_empty_page(self):
        """SAP's page requires JS; an empty parse should reliably trigger fallback."""
        result = sap.discover_filings(_mock_session_empty_page())
        assert result == sap.FALLBACK_FILINGS

    def test_fallback_on_connection_error(self):
        result = sap.discover_filings(_mock_session_raises(requests.ConnectionError()))
        assert result == sap.FALLBACK_FILINGS


# ---------------------------------------------------------------------------
# Discovery pipeline (DB integration)
# ---------------------------------------------------------------------------

class TestRunDiscovery:
    def _patch_session(self):
        """Patch _make_session so all adapters fall back to hardcoded URLs."""
        return patch(
            "fiscal.discovery._make_session",
            return_value=_mock_session_raises(requests.ConnectionError("mocked")),
        )

    def test_inserts_filings_for_all_companies(self, tmp_db):
        from fiscal import discovery

        with self._patch_session():
            results = discovery.run_discovery(db_path=tmp_db)

        assert set(results.keys()) == {"adyen", "heineken", "sap"}
        for key, count in results.items():
            assert count > 0, f"Expected new filings for {key!r}"

        with db.get_connection(tmp_db) as conn:
            rows = db.get_filings(conn)
        assert len(rows) > 0

    def test_inserts_only_for_specified_company(self, tmp_db):
        from fiscal import discovery

        with self._patch_session():
            results = discovery.run_discovery(company="adyen", db_path=tmp_db)

        assert list(results.keys()) == ["adyen"]
        with db.get_connection(tmp_db) as conn:
            adyen_rows = db.get_filings(conn, company="adyen")
            heineken_rows = db.get_filings(conn, company="heineken")
        assert len(adyen_rows) > 0
        assert len(heineken_rows) == 0

    def test_second_run_inserts_zero_new_filings(self, tmp_db):
        from fiscal import discovery

        with self._patch_session():
            first = discovery.run_discovery(company="adyen", db_path=tmp_db)
            second = discovery.run_discovery(company="adyen", db_path=tmp_db)

        assert second["adyen"] == 0, "Duplicate URLs must be skipped"

    def test_annual_reports_are_classified_correctly(self, tmp_db):
        from fiscal import discovery

        with self._patch_session():
            discovery.run_discovery(company="adyen", db_path=tmp_db)

        with db.get_connection(tmp_db) as conn:
            annual = db.get_filings(conn, company="adyen", document_type="annual_report")

        assert len(annual) > 0, "Adyen fallback should produce annual_report filings"

    def test_fiscal_years_are_set(self, tmp_db):
        from fiscal import discovery

        with self._patch_session():
            discovery.run_discovery(company="adyen", db_path=tmp_db)

        with db.get_connection(tmp_db) as conn:
            rows = db.get_filings(conn, company="adyen", document_type="annual_report")

        years_set = [r for r in rows if r["fiscal_year"] is not None]
        assert len(years_set) > 0, "At least some filings should have a fiscal_year"

    def test_invalid_company_raises_value_error(self, tmp_db):
        from fiscal import discovery

        with pytest.raises(ValueError, match="Unknown company key"):
            discovery.run_discovery(company="INVALID", db_path=tmp_db)
