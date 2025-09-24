import os
import sys
import time
import json

from preprocess import pdf_to_images, deskew_image, is_blank_image
from ocr import ocr_pages
from llm_query import LLMClient, extract_structured


def run_pipeline(input_path, outdir='output', model_path=None):
    os.makedirs(outdir, exist_ok=True)
    ts = int(time.time())
    report_path = os.path.join(outdir, f'{ts}_report.json')

    # 1) Convert PDF -> images (if already images, expand to list)
    if input_path.lower().endswith('.pdf'):
        image_paths = pdf_to_images(input_path, output_dir='processed_images')
    else:
        # assume single image file
        image_paths = [input_path]

    kept = []
    # 2) Deskew, remove blank pages
    for p in image_paths:
        deskewed = deskew_image(p, out_path=p.replace('page', 'deskewed'))
        if is_blank_image(deskewed):
            print('Dropping blank page', deskewed)
            continue
        kept.append(deskewed)

    # 3) OCR
    ocr_results = ocr_pages(kept, out_txt_dir='ocr_text')

    # quick concatenation with page markers
    combined = ''
    for p, txt in ocr_results:
        page_no = os.path.splitext(os.path.basename(p))[0]
        combined += f'=== PAGE: {page_no} ===\n'
        combined += txt + '\n\n'

    # 4) LLM extraction
    llm = LLMClient(model_name_or_path=model_path)
    structured = extract_structured(llm, combined)

    report = {
        'input': input_path,
        'pages_processed': len(kept),
        'structured': structured,
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('Report written to', report_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python pipeline.py path/to/input.pdf output_dir [optional_model_path]')
        sys.exit(1)

    input_path = sys.argv[1]
    outdir = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    run_pipeline(input_path, outdir, model_path)
