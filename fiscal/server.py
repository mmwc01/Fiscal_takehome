"""
Flask web server. Exposes a single-page UI with per-company buttons that run
the LLM pipeline and return a PDF for that company.

Start with: fiscal serve [--port 8080]
"""
import logging
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, send_file

load_dotenv()

from fiscal.ir_scraper import run_ir_scrape
from fiscal.llm.client import LLMClient
from fiscal.llm.field_value_extractor import run_field_extraction
from fiscal.llm_pdf_builder import build_pdf_from_values
from fiscal.scripts.run_company_field_discovery_from_pdfs import run as run_discovery

logger = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_OUTPUT_DIR    = str(_PROJECT_ROOT / "output")
_CATALOG_DIR   = _PROJECT_ROOT / "data" / "company_field_catalogs"
_PDF_BASE_DIR  = _PROJECT_ROOT / "data" / "pdfs"
_VALUES_DIR    = _PROJECT_ROOT / "data" / "company_field_values"



COMPANIES = ["adyen", "heineken", "sap"]

app = Flask(__name__)

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fiscal Pipeline</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   Helvetica, Arial, sans-serif;
      background: #F4F6F9;
      color: #1A1A2E;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 32px;
      padding: 40px 20px;
    }

    .card {
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 2px 16px rgba(0,0,0,0.08);
      padding: 40px 48px;
      width: 100%;
      max-width: 640px;
      text-align: center;
    }

    h1 {
      font-size: 1.6rem;
      font-weight: 700;
      margin-bottom: 8px;
      color: #1A1A2E;
    }

    .subtitle {
      font-size: 0.9rem;
      color: #666;
      margin-bottom: 32px;
      line-height: 1.5;
    }

    .btn-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .run-btn {
      background: #0891B2;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      padding: 14px 36px;
      cursor: pointer;
      transition: background 0.15s, transform 0.1s;
      width: 100%;
    }
    .run-btn:hover:not(:disabled) { background: #0E7490; }
    .run-btn:active:not(:disabled) { transform: scale(0.98); }
    .run-btn:disabled { background: #CBD5E1; cursor: not-allowed; }

    #status {
      margin-top: 20px;
      font-size: 0.85rem;
      color: #555;
      min-height: 22px;
      line-height: 1.5;
    }

    .success { color: #059669; font-weight: 600; }
    .error   { color: #DC2626; font-weight: 600; }

    .spinner {
      display: inline-block;
      width: 14px; height: 14px;
      border: 2px solid #A5B4FC;
      border-top-color: #4338CA;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="card">
    <h1>Fiscal Pipeline</h1>
    <p class="subtitle">
      Select a company to generate and download its financial report.
    </p>

    <div class="btn-group">
      <button class="run-btn" onclick="runPipeline('adyen',    this)">Adyen — Download PDF</button>
      <button class="run-btn" onclick="runPipeline('heineken', this)">Heineken — Download PDF</button>
      <button class="run-btn" onclick="runPipeline('sap',      this)">SAP — Download PDF</button>
    </div>

    <div id="status"></div>
  </div>

  <script>
    async function runPipeline(company, btn) {
      const allBtns = document.querySelectorAll('.run-btn');
      const status  = document.getElementById('status');
      const label   = company.charAt(0).toUpperCase() + company.slice(1);

      allBtns.forEach(b => b.disabled = true);
      status.innerHTML = '<span class="spinner"></span> Generating report for '
                       + label + '… (this may take a few minutes)';

      try {
        const resp = await fetch('/run-llm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ company }),
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ error: resp.statusText }));
          status.innerHTML = '<span class="error">Pipeline failed.</span> '
                           + (err.error || resp.statusText);
          return;
        }

        const blob = await resp.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = resp.headers.get('X-Filename') || company + '_report.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);

        status.innerHTML = '<span class="success">Done!</span> PDF downloaded.';

      } catch (e) {
        status.innerHTML = '<span class="error">Network error:</span> ' + e.message;
      } finally {
        allBtns.forEach(b => b.disabled = false);
      }
    }
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(_HTML)



@app.post("/run-llm")
def run_llm():
    body    = request.get_json(silent=True) or {}
    company = body.get("company", "").strip().lower()

    if company not in COMPANIES:
        return jsonify({"error": f"Unknown company {company!r}. Must be one of {COMPANIES}"}), 400

    client = LLMClient()
    if not client.enabled():
        return jsonify({"error": "OPENAI_API_KEY is not set — check your .env file"}), 500

    try:
        # Step 0: scrape IR website for new PDFs (skips already-downloaded files)
        logger.info("[server/llm] %s — checking IR website for new PDFs", company)
        try:
            scrape_result = run_ir_scrape(company, pdf_base_dir=_PDF_BASE_DIR)
            if scrape_result["downloaded"]:
                logger.info(
                    "[server/llm] %s — downloaded %d new PDF(s) from IR site",
                    company, scrape_result["downloaded"],
                )
        except Exception as scrape_exc:
            # Non-fatal: proceed with whatever PDFs are already present
            logger.warning("[server/llm] %s — IR scrape failed (continuing): %s", company, scrape_exc)

        # Step 1: run field discovery (always refresh the catalog)
        logger.info("[server/llm] %s — running field discovery", company)
        catalog_path = run_discovery(
            company=company,
            base_dir=str(_PDF_BASE_DIR),
        )

        # Step 2: extract values from all PDFs
        values_output = _VALUES_DIR / f"{company}.json"
        values_path = run_field_extraction(
            company,
            catalog_path=catalog_path,
            pdf_dir=_PDF_BASE_DIR / company,
            client=client,
            output_path=values_output,
        )

        # Step 3: build PDF
        out_dir = Path(_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = build_pdf_from_values(values_path, output_dir=out_dir)

        if not pdf_path.exists():
            return jsonify({"error": "No year data found in any PDF — extraction produced no values. Check that PDFs exist and pages are being detected."}), 500

        filename = pdf_path.name
        return send_file(
            str(pdf_path),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
            max_age=0,
        ), 200, {"X-Filename": filename}

    except Exception as exc:
        logger.exception("LLM pipeline error")
        return jsonify({"error": str(exc)}), 500


def run_server(host: str = "127.0.0.1", port: int = 8080, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server(debug=True)
