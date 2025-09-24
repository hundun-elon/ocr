# OCR + LLM Compliance Pipeline

local pipeline that processes scanned documents and extracts structured information for compliance checks, combines Tesseract OCR for text extraction and a local Mistral model for natural language queries.

## Features
- Preprocess scanned PDFs/images (deskew, remove blank pages).
- Perform OCR using Tesseract.
- Query extracted text with a local LLM (Mistral).
- Generate structured JSON reports.

## Requirements
- Python 3.9+
- Tesseract OCR installed
- Dependencies from `requirements.txt`

## Setup
```bash
pip install -r requirements.txt
