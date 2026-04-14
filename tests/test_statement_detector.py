"""
Tests for fiscal/extraction/statement_detector.py

Organised into:
  TestPageTriggers       — each statement type detected via heading phrases
  TestTableAnchors       — anchor-only detection (no page trigger present)
  TestConfidenceLevels   — verify the confidence field is set correctly
  TestUnknown            — inputs that should return "unknown"
  TestAmbiguousInput     — pages that trigger multiple types
  TestEdgeCases          — whitespace, case, word-boundary, notes prose
"""
import pytest
from fiscal.extraction.statement_detector import detect, DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type(text: str) -> str:
    return detect(text).statement_type

def _confidence(text: str) -> str:
    return detect(text).confidence

def _reason(text: str) -> str:
    return detect(text).reason


# ---------------------------------------------------------------------------
# PAGE_TRIGGERS — income statement
# ---------------------------------------------------------------------------

class TestIncomeStatementTriggers:
    def test_income_statement_heading(self):
        assert _type("Consolidated Income Statement") == "income_statement"

    def test_income_statement_lowercase(self):
        assert _type("income statement") == "income_statement"

    def test_income_statement_uppercase(self):
        assert _type("CONSOLIDATED INCOME STATEMENT") == "income_statement"

    def test_profit_and_loss(self):
        assert _type("Profit and Loss Account") == "income_statement"

    def test_profit_ampersand_loss(self):
        assert _type("Profit & Loss") == "income_statement"

    def test_statement_of_income(self):
        assert _type("Statement of Income\nYear ended December 31, 2023") == "income_statement"

    def test_statement_of_comprehensive_income(self):
        assert _type(
            "Consolidated Statement of Comprehensive Income"
        ) == "income_statement"

    def test_statement_of_profit(self):
        assert _type("Statement of Profit\nfor the year ended 31 December 2022") == "income_statement"

    def test_consolidated_income(self):
        assert _type("Consolidated Income\n(in € millions)") == "income_statement"

    def test_results_from_operations(self):
        assert _type("Results from Operations\n2023    2022") == "income_statement"

    def test_multiline_page_with_trigger(self):
        page = (
            "Adyen N.V.\n"
            "Consolidated Income Statement\n"
            "for the year ended 31 December 2023\n"
            "                     2023      2022\n"
            "Revenue              1,234     1,100\n"
        )
        assert _type(page) == "income_statement"


# ---------------------------------------------------------------------------
# PAGE_TRIGGERS — balance sheet
# ---------------------------------------------------------------------------

class TestBalanceSheetTriggers:
    def test_balance_sheet_heading(self):
        assert _type("Consolidated Balance Sheet") == "balance_sheet"

    def test_balance_sheet_lowercase(self):
        assert _type("balance sheet") == "balance_sheet"

    def test_statement_of_financial_position(self):
        assert _type(
            "Consolidated Statement of Financial Position\nas at 31 December 2023"
        ) == "balance_sheet"

    def test_consolidated_balance(self):
        assert _type("Consolidated Balance\n31 December 2023") == "balance_sheet"

    def test_multiline_page_with_trigger(self):
        page = (
            "SAP SE\n"
            "Consolidated Balance Sheet\n"
            "as of December 31, 2023\n"
            "                           2023      2022\n"
            "Total assets             65,432    61,234\n"
        )
        assert _type(page) == "balance_sheet"


# ---------------------------------------------------------------------------
# PAGE_TRIGGERS — cash flow statement
# ---------------------------------------------------------------------------

class TestCashFlowTriggers:
    def test_cash_flow_statement_heading(self):
        assert _type("Consolidated Cash Flow Statement") == "cash_flow_statement"

    def test_statement_of_cash_flows(self):
        assert _type("Statement of Cash Flows") == "cash_flow_statement"

    def test_statement_of_cash_flow_singular(self):
        assert _type("Statement of Cash Flow") == "cash_flow_statement"

    def test_cash_flows_from(self):
        assert _type("Cash flows from operating activities") == "cash_flow_statement"

    def test_cash_flow_lowercase(self):
        assert _type("cash flow statement") == "cash_flow_statement"

    def test_multiline_page_with_trigger(self):
        page = (
            "Heineken N.V.\n"
            "Consolidated Statement of Cash Flows\n"
            "Year ended 31 December 2023\n"
            "                              2023     2022\n"
            "Cash flows from operations   1,456    1,289\n"
        )
        assert _type(page) == "cash_flow_statement"


# ---------------------------------------------------------------------------
# TABLE_ANCHORS — no page trigger present
# ---------------------------------------------------------------------------

class TestTableAnchors:
    def test_income_anchors_revenue_and_gross_profit(self):
        text = "Revenue\n1,234\nGross profit\n567\nOperating expenses\n400"
        assert _type(text) == "income_statement"

    def test_income_anchors_ebitda_and_net_income(self):
        text = "EBITDA    890\nNet income    234"
        assert _type(text) == "income_statement"

    def test_income_anchors_operating_profit(self):
        text = "Revenue    1,000\nOperating profit    250"
        assert _type(text) == "income_statement"

    def test_balance_anchors_total_assets_and_equity(self):
        text = "Total assets    65,432\nTotal equity    34,567"
        assert _type(text) == "balance_sheet"

    def test_balance_anchors_assets_and_liabilities(self):
        text = "Total assets    65,432\nTotal liabilities    30,865"
        assert _type(text) == "balance_sheet"

    def test_balance_anchors_current_assets_and_equity(self):
        text = "Current assets    12,345\nShareholders' equity    34,567"
        assert _type(text) == "balance_sheet"

    def test_cashflow_anchors_operating_and_investing(self):
        text = "Operating activities    1,456\nInvesting activities    (234)"
        assert _type(text) == "cash_flow_statement"

    def test_cashflow_anchors_all_three_activities(self):
        text = (
            "Operating activities    1,456\n"
            "Investing activities    (234)\n"
            "Financing activities    (800)\n"
        )
        assert _type(text) == "cash_flow_statement"

    def test_cashflow_anchors_net_cash_and_activities(self):
        text = "Net cash from operating activities    1,200\nInvesting activities    (300)"
        assert _type(text) == "cash_flow_statement"

    def test_single_anchor_term_returns_unknown(self):
        # "revenue" alone appears in prose everywhere — not enough signal
        assert _type("Revenue recognition is covered in Note 3.") == "unknown"

    def test_single_balance_anchor_insufficient(self):
        # one term is not enough
        assert _type("Total assets were up 5% year-over-year.") == "unknown"


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------

class TestConfidenceLevels:
    def test_page_trigger_gives_high_confidence(self):
        assert _confidence("Consolidated Income Statement") == "high"

    def test_balance_page_trigger_high(self):
        assert _confidence("Statement of Financial Position") == "high"

    def test_cashflow_page_trigger_high(self):
        assert _confidence("Statement of Cash Flows") == "high"

    def test_anchor_only_gives_medium_confidence(self):
        text = "Total assets    65,432\nTotal equity    34,567"
        assert _confidence(text) == "medium"

    def test_no_match_gives_low_confidence(self):
        assert _confidence("The board of directors met on January 15.") == "low"

    def test_unknown_type_gives_low_confidence(self):
        result = detect("Some random text with no financial keywords.")
        assert result.confidence == "low"
        assert result.statement_type == "unknown"


# ---------------------------------------------------------------------------
# Unknown — inputs that should not classify
# ---------------------------------------------------------------------------

class TestUnknown:
    def test_empty_string(self):
        assert _type("") == "unknown"

    def test_whitespace_only(self):
        assert _type("   \n\t  ") == "unknown"

    def test_random_prose(self):
        assert _type("The board of directors met on January 15, 2024.") == "unknown"

    def test_table_of_contents_page(self):
        # ToC mentions all sections but with page numbers — should still trigger
        # (filtering ToC pages is the pipeline's job, not the detector's)
        text = (
            "Table of Contents\n"
            "Consolidated Income Statement ......... 45\n"
            "Consolidated Balance Sheet ............. 47\n"
            "Cash Flow Statement .................... 49\n"
        )
        # Multiple triggers → ambiguous but still classified, not unknown
        result = detect(text)
        assert result.statement_type != "unknown"

    def test_company_overview_page(self):
        text = (
            "About Adyen\n"
            "Adyen is a global payment company founded in 2006.\n"
            "We processed over €1 trillion in payments in 2023.\n"
        )
        assert _type(text) == "unknown"

    def test_auditor_report_page(self):
        text = (
            "Independent Auditor's Report\n"
            "To the shareholders of Heineken N.V.\n"
            "We have audited the accompanying financial statements.\n"
        )
        assert _type(text) == "unknown"


# ---------------------------------------------------------------------------
# Ambiguous input — multiple statement types on one page
# ---------------------------------------------------------------------------

class TestAmbiguousInput:
    def test_two_triggers_picks_more_matches(self):
        # income_statement has 2 triggers; balance_sheet has 1 → income_statement wins
        text = (
            "Consolidated Income Statement\n"
            "Statement of Comprehensive Income\n"
            "Balance Sheet\n"
        )
        result = detect(text)
        assert result.statement_type == "income_statement"
        assert result.confidence == "high"

    def test_ambiguity_noted_in_reason(self):
        text = "Income Statement\nBalance Sheet"
        result = detect(text)
        assert "ambiguous" in result.reason.lower() or "also triggered" in result.reason

    def test_toc_still_returns_a_type(self):
        # A ToC page is ambiguous; we don't expect unknown, we expect a best guess
        text = "Income Statement 45\nBalance Sheet 47\nCash Flow Statement 49"
        result = detect(text)
        assert result.statement_type != "unknown"
        assert result.confidence == "high"


# ---------------------------------------------------------------------------
# Edge cases — word boundaries, case, whitespace variants, notes prose
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_case_insensitive_trigger(self):
        assert _type("INCOME STATEMENT") == "income_statement"
        assert _type("income statement") == "income_statement"
        assert _type("Income Statement") == "income_statement"

    def test_extra_whitespace_in_trigger(self):
        # "balance  sheet" (double space) — \s+ handles this
        assert _type("balance  sheet") == "balance_sheet"

    def test_word_boundary_no_partial_match(self):
        # "incomestatement" is not a valid heading — should not match
        assert _type("incomestatement totals for the year") == "unknown"

    def test_notes_mention_of_income_statement_triggers(self):
        # A notes page may contain "income statement" in a sentence —
        # this WILL match as high confidence (filtering notes is pipeline's job)
        text = (
            "Note 3: Revenue Recognition\n"
            "Revenue is presented in the income statement net of VAT.\n"
        )
        result = detect(text)
        # We accept this as a known false-positive at the detector level;
        # the pipeline's table-shape check will reject notes tables downstream.
        assert result.statement_type == "income_statement"
        assert result.confidence == "high"

    def test_single_revenue_word_is_insufficient(self):
        assert _type("Revenue") == "unknown"

    def test_anchor_requires_two_terms_not_one(self):
        # "Gross profit" is one anchor term for income_statement — not enough alone
        assert _type("Gross profit was higher than expected this year.") == "unknown"

    def test_reason_string_is_non_empty(self):
        for text in [
            "Income Statement",
            "Total assets\nTotal equity",
            "Some random text",
        ]:
            result = detect(text)
            assert result.reason, f"Empty reason for input: {text!r}"

    def test_reason_contains_matched_text_for_trigger(self):
        result = detect("Consolidated Balance Sheet")
        assert "balance sheet" in result.reason.lower()

    def test_reason_contains_matched_terms_for_anchor(self):
        result = detect("Total assets    100\nTotal equity    50")
        assert "total assets" in result.reason.lower() or "total equity" in result.reason.lower()

    def test_shareholders_equity_apostrophe_variants(self):
        assert _type("Shareholders' equity    34,567\nTotal assets    65,432") == "balance_sheet"
        assert _type("Shareholders\u2019 equity    34,567\nTotal assets    65,432") == "balance_sheet"

    def test_result_is_frozen_dataclass(self):
        result = detect("Income Statement")
        with pytest.raises((AttributeError, TypeError)):
            result.statement_type = "other"  # type: ignore[misc]
