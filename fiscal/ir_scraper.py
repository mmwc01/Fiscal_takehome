"""
Investor Relations PDF scraper.

Scrapes each company's IR annual-reports page, finds PDF links, and downloads
any that aren't already present in data/pdfs/<company>/.

Hardcoded company config: name, ticker, IR page URL, and a URL pattern that
identifies annual report PDFs (to filter out other documents on the same page).

Entry point: run_ir_scrape(company, pdf_base_dir, dry_run=False)
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Company config
# ---------------------------------------------------------------------------

COMPANY_INFO: dict[str, dict] = {
    "adyen": {
        "name": "Adyen N.V.",
        "ticker": "ADYEN",
        "exchange": "Euronext Amsterdam",
        "ir_url": "https://investors.adyen.com/financial-results/annual-reports",
        # PDF href must match this pattern (case-insensitive)
        "pdf_pattern": re.compile(r"annual.?report", re.IGNORECASE),
    },
    "heineken": {
        "name": "Heineken N.V.",
        "ticker": "HEIA",
        "exchange": "Euronext Amsterdam",
        "ir_url": "https://www.theheinekencompany.com/investors/reports-and-presentations/annual-reports",
        "pdf_pattern": re.compile(r"annual.?report", re.IGNORECASE),
    },
    "sap": {
        "name": "SAP SE",
        "ticker": "SAP",
        "exchange": "Frankfurt Stock Exchange / NYSE",
        "ir_url": "https://www.sap.com/investors/en/reports.html",
        "pdf_pattern": re.compile(r"integrated.?report|annual.?report", re.IGNORECASE),
    },
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; FiscalPipeline/1.0; "
        "+https://github.com/fiscal-pipeline)"
    )
}
_REQUEST_TIMEOUT = 20
_DELAY_BETWEEN_DOWNLOADS = 1.5  # seconds — be polite to IR servers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.warning("[ir_scraper] fetch failed %s: %s", url, exc)
        return None


def _find_pdf_links(html: str, base_url: str, pattern: re.Pattern) -> list[str]:
    """
    Return absolute PDF URLs whose href or surrounding text matches pattern.
    """
    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []
    seen: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        if not href.lower().endswith(".pdf"):
            continue
        abs_url = urljoin(base_url, href)
        if abs_url in seen:
            continue
        # Match against the href path or the link text
        link_text = tag.get_text(" ", strip=True)
        if pattern.search(href) or pattern.search(link_text):
            found.append(abs_url)
            seen.add(abs_url)

    return found


def _year_from_url(url: str) -> int | None:
    m = re.search(r"(20\d{2})", urlparse(url).path)
    return int(m.group(1)) if m else None


def _filename_for(url: str, company: str) -> str:
    """Derive a clean local filename from the PDF URL."""
    name = Path(urlparse(url).path).name
    # If the filename doesn't contain the year, try to add it
    if not re.search(r"20\d{2}", name):
        year = _year_from_url(url)
        if year:
            stem, suffix = name.rsplit(".", 1) if "." in name else (name, "pdf")
            name = f"{stem}_{year}.{suffix}"
    return name


def _download_pdf(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("[ir_scraper] downloaded %s → %s", url, dest.name)
        return True
    except requests.RequestException as exc:
        logger.warning("[ir_scraper] download failed %s: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ir_scrape(
    company: str,
    pdf_base_dir: str | Path = "data/pdfs",
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Scrape the company's IR page and download any new annual report PDFs.

    Returns a dict with keys: found, downloaded, skipped, failed.
    """
    info = COMPANY_INFO.get(company)
    if not info:
        raise ValueError(
            f"Unknown company {company!r}. Must be one of {list(COMPANY_INFO)}"
        )

    ir_url = info["ir_url"]
    pattern = info["pdf_pattern"]
    dest_dir = Path(pdf_base_dir) / company
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[ir_scraper] %s — scraping %s", company, ir_url)

    html = _fetch(ir_url)
    if not html:
        logger.error("[ir_scraper] %s — could not fetch IR page", company)
        return {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0}

    pdf_urls = _find_pdf_links(html, ir_url, pattern)
    logger.info("[ir_scraper] %s — found %d PDF link(s)", company, len(pdf_urls))

    downloaded = skipped = failed = 0

    for url in pdf_urls:
        filename = _filename_for(url, company)
        dest = dest_dir / filename

        if dest.exists():
            logger.debug("[ir_scraper] %s — already have %s, skipping", company, filename)
            skipped += 1
            continue

        if dry_run:
            logger.info("[ir_scraper] %s — DRY RUN would download %s", company, url)
            skipped += 1
            continue

        success = _download_pdf(url, dest)
        if success:
            downloaded += 1
        else:
            failed += 1

        time.sleep(_DELAY_BETWEEN_DOWNLOADS)

    return {
        "found": len(pdf_urls),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
