"""Tests for fiscal.classification — document type and fiscal year inference."""
import pytest
from fiscal.classification import classify_filing, _extract_year


class TestDocumentType:
    def test_annual_report_title(self):
        doc_type, _ = classify_filing("Annual Report 2022", "https://example.com/ar.pdf")
        assert doc_type == "annual_report"

    def test_annual_report_from_url(self):
        doc_type, _ = classify_filing(
            "Download",
            "https://media.adyen.com/documents/adyen-annual-report-2023.pdf",
        )
        assert doc_type == "annual_report"

    def test_annual_review(self):
        doc_type, _ = classify_filing("Annual Review 2021", "https://example.com/r.pdf")
        assert doc_type == "annual_report"

    def test_dutch_jaarverslag(self):
        doc_type, _ = classify_filing("Jaarverslag 2021", "https://example.com/j.pdf")
        assert doc_type == "annual_report"

    def test_german_geschaeftsbericht(self):
        doc_type, _ = classify_filing(
            "SAP Geschäftsbericht 2022", "https://example.com/g.pdf"
        )
        assert doc_type == "annual_report"

    def test_german_without_umlaut(self):
        doc_type, _ = classify_filing(
            "SAP Geschaftsbericht 2022", "https://example.com/g.pdf"
        )
        assert doc_type == "annual_report"

    def test_sustainability_overrides_annual(self):
        """Sustainability report that mentions 'annual' in title → sustainability wins."""
        doc_type, _ = classify_filing(
            "Annual Sustainability Report 2022", "https://example.com/sr.pdf"
        )
        assert doc_type == "sustainability"

    def test_esg_report(self):
        doc_type, _ = classify_filing("ESG Report 2022", "https://example.com/esg.pdf")
        assert doc_type == "sustainability"

    def test_quarterly(self):
        doc_type, _ = classify_filing("Q3 2022 Results", "https://example.com/q3.pdf")
        assert doc_type == "quarterly"

    def test_quarterly_from_url(self):
        doc_type, _ = classify_filing("Results", "https://example.com/q1-2022.pdf")
        assert doc_type == "quarterly"

    def test_interim_half_year(self):
        doc_type, _ = classify_filing(
            "Half-Year Report 2022", "https://example.com/h1.pdf"
        )
        assert doc_type == "interim_report"

    def test_interim_keyword(self):
        doc_type, _ = classify_filing(
            "Interim Results 2022", "https://example.com/interim.pdf"
        )
        assert doc_type == "interim_report"

    def test_other_document(self):
        doc_type, _ = classify_filing(
            "Investor Day Presentation", "https://example.com/ir.pdf"
        )
        assert doc_type == "other"

    def test_proxy_is_other(self):
        doc_type, _ = classify_filing(
            "Notice of AGM 2023", "https://example.com/agm.pdf"
        )
        assert doc_type == "other"


class TestFiscalYearInference:
    def test_year_from_title(self):
        _, year = classify_filing("Annual Report 2019", "https://example.com/ar.pdf")
        assert year == 2019

    def test_year_from_url(self):
        _, year = classify_filing(
            "Annual Report", "https://example.com/annual-report-2021.pdf"
        )
        assert year == 2021

    def test_year_in_title_preferred_over_url(self):
        # Both have years; title year (2023) should be found first
        _, year = classify_filing(
            "Annual Report 2023",
            "https://example.com/reports/2020/ar.pdf",
        )
        assert year == 2023

    def test_no_year_returns_none(self):
        _, year = classify_filing("Annual Report", "https://example.com/ar.pdf")
        assert year is None

    def test_four_digit_year_only(self):
        """Should not match years outside 2000–2029."""
        _, year = classify_filing("Report 1999", "https://example.com/ar.pdf")
        assert year is None


class TestExtractYear:
    def test_found(self):
        assert _extract_year("Annual Report 2022") == 2022

    def test_not_found(self):
        assert _extract_year("No year here") is None

    def test_returns_first_match(self):
        assert _extract_year("2021 and 2022") == 2021
