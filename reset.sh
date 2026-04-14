#!/usr/bin/env bash
# reset.sh — wipe all generated/cached data so the pipeline starts completely fresh.
#
# Safe to run at any time. Does NOT delete source PDFs in data/pdfs/.
# Does NOT touch .env or any source code.
#
# Usage:
#   bash reset.sh

set -euo pipefail

echo "Clearing Fiscal pipeline caches and outputs..."

# Page classification cache (pdfplumber results keyed by SHA-256)
rm -rf data/pdf_cache/
echo "  ✓ data/pdf_cache/"

# Field catalogs produced by discovery step
rm -rf data/company_field_catalogs/
echo "  ✓ data/company_field_catalogs/"

# Extracted values JSON produced by extraction step
rm -rf data/company_field_values/
echo "  ✓ data/company_field_values/"

# Generated PDF reports
rm -rf output/
echo "  ✓ output/"

echo ""
echo "Done. Run 'fiscal serve' to start fresh."
