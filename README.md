# Fiscal-Takehome-Pipeline

LLM-powered pipeline that reads 10 years of annual report PDFs for Adyen, Heineken,
and SAP, extracts financial statement data, and serves a web UI that generates a
formatted PDF report on demand.

## Quick start

```bash
source .venv/bin/activate
cp .env.example .env        # add your OPENAI_API_KEY
fiscal serve                # → http://localhost:8080/
```

Click a company button in the browser. The pipeline runs end-to-end and downloads
the PDF report automatically.

## Setup from scratch

```bash
# 1. Create virtual environment and install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Add your OpenAI API key
cp .env.example .env             # then edit .env and set OPENAI_API_KEY=sk-...

# 3. Place annual report PDFs in data/pdfs/<company>/
#    e.g. data/pdfs/adyen/adyen_annual_report_2024.pdf

# 4. Start the web UI
fiscal serve
```

## How the pipeline works

Each button click runs three steps automatically:

### Step 1 — Field discovery
One LLM call per statement type (income statement, balance sheet, cash flow) on the
most recent PDF. Produces a **field catalog** — the normalized list of financial line
items to extract across all years. Saved to `data/company_field_catalogs/`.

### Step 2 — Value extraction
For each PDF (up to 10 years), one LLM call per statement type reads both the raw
text and structured table views of the financial pages and returns values for every
catalog field. All calls run in parallel. Results saved to `data/company_field_values/`.

Key extraction details:
- Only high-confidence statement pages (and their immediate neighbors) are sent to
  the LLM — narrative pages are excluded to keep prompts small
- If a prompt exceeds ~200K chars, it is automatically chunked across multiple calls
- PDF page content is cached by SHA-256 hash — reruns skip pdfplumber entirely for
  unchanged PDFs (`data/pdf_cache/`)

### Step 3 — PDF report
Written to `output/<company>_llm_report.pdf` and streamed to the browser.

## Web UI

`fiscal serve` starts a Flask server on `http://127.0.0.1:8080`.

| Button | What happens |
|--------|-------------|
| Adyen — Download PDF | Field discovery → extraction → PDF for Adyen |
| Heineken — Download PDF | Field discovery → extraction → PDF for Heineken |
| SAP — Download PDF | Field discovery → extraction → PDF for SAP |

Discovery always re-runs on button click to pick up the latest available PDF.
Extraction uses cached page content where available.

## CLI scripts

Run individual pipeline steps directly with the following script:

```bash
# Field discovery (generates catalog from latest PDF)
python -m fiscal.scripts.run_company_field_discovery_from_pdfs --company adyen

# Value extraction + PDF report (uses existing catalog)
python -m fiscal.scripts.run_field_extraction --company adyen

# Options
python -m fiscal.scripts.run_field_extraction --company adyen --no-pdf
python -m fiscal.scripts.run_field_extraction --company adyen --max-pdfs 5
python -m fiscal.scripts.run_field_extraction --company adyen --write-db
```

## Resetting

To wipe all cached and generated data and start completely fresh:

```bash
bash reset.sh
```

This clears `data/pdf_cache/`, `data/company_field_catalogs/`, `data/company_field_values/`,
and `output/`. Source PDFs in `data/pdfs/` are left untouched.

## Running tests

```bash
pytest
pytest -v
pytest tests/test_classifier.py
```
