#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


def _add_project_root_to_path() -> None:
    here = Path(__file__).resolve()
    root = here.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    _add_project_root_to_path()

    import pdfplumber
    from backend.app.adapters.content.pdf import PDFExtractor, _build_text_from_words, _fix_pdf_urls
    from backend.app.config.loader import load_config
    from backend.app.util.text_processing import clean_document_text

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to a PDF file")
    args = ap.parse_args()

    config = load_config("backend/config.yaml")
    extractor = PDFExtractor()

    pdf_path = Path(args.path).expanduser().resolve()
    uri = f"file://{pdf_path}"

    content = extractor.extract(uri)

    print("=== PDFExtractor ===")
    print("=== TITLE ===")
    print(content.title or "(none)")
    print()

    print("=== TEXT (first 6000 chars) ===")
    text = content.text or ""
    print(text[:6000])
    print()

    sections = content.extra.get("sections") if isinstance(content.extra, dict) else None
    print("=== SECTIONS ===")
    if sections:
        for s in sections:
            title = s.get("title")
            level = s.get("level")
            start = s.get("start")
            end = s.get("end")
            print(f"- level={level} start={start} end={end} title={title!r}")
    else:
        print("(no sections)")

    print("=== pdfplumber ===")
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = pdf.pages[:2]
            for idx, page in enumerate(pages, start=1):
                page_text = _build_text_from_words(page) or ""
                cleaned = clean_document_text(page_text)
                cleaned = _fix_pdf_urls(cleaned)
                print()
                print(f"=== PAGE {idx} RAW from pdfplumber and _build_text_from_words ===")
                print(page_text)
                print()
                print(f"=== PAGE {idx} CLEANED after clean_document_text and _fix_pdf_urls ===")
                print(cleaned)
    except Exception as e:
        print()
        print(f"[warn] failed to dump per-page text: {e}")


if __name__ == "__main__":
    main()
