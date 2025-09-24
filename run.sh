#!/usr/bin/env bash
set -e
INFILE="$1"
if [ -z "$INFILE" ]; then
echo "Usage: $0 path/to/scanned.pdf"
exit 1
fi

mkdir -p input output processed_images ocr_text
cp "$INFILE" input/
python pipeline.py input/$(basename "$INFILE") output/


echo "Done. Check the output/ folder for the report and intermediate files."