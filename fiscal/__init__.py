"""
fiscal — LLM-powered financial report pipeline.

Pipeline steps (triggered via web UI or CLI scripts):
  1. Field discovery  — LLM reads the latest PDF, produces a field catalog
  2. Value extraction — LLM extracts values for each catalog field across all PDFs
  3. PDF report       — Pivots extracted values into a formatted landscape A4 PDF
"""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
