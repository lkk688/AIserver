from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pymupdf
from bs4 import BeautifulSoup
from pypdf import PdfReader

USER_AGENT = "ia-phase1-parser/0.1 (+https://example.local)"
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _default_pdf_dir() -> Path:
    configured = (Path.cwd() / ".ia_phase1_data" / "pdfs")
    configured.mkdir(parents=True, exist_ok=True)
    return configured


def _safe_filename(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"{h}.pdf"


def _sanitize_extracted_text(value: Any) -> str:
    text = str(value or "").replace("\x00", "")
    return _SURROGATE_RE.sub("\uFFFD", text)


def _looks_like_pdf_bytes(data: bytes) -> bool:
    head = data[:1024].lstrip()
    return b"%PDF-" in head


def describe_google_drive_source(value: str) -> Optional[Dict[str, str]]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path_parts = [part for part in (parsed.path or "").split("/") if part]
    query = parse_qs(parsed.query or "")

    def _build_doc_export(kind: str, file_id: str) -> Dict[str, str]:
        if kind == "document":
            return {
                "source_kind": "google_doc_export",
                "file_id": file_id,
                "download_url": f"https://docs.google.com/document/d/{file_id}/export?format=pdf",
            }
        if kind == "spreadsheets":
            return {
                "source_kind": "google_sheet_export",
                "file_id": file_id,
                "download_url": f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=pdf",
            }
        if kind == "presentation":
            return {
                "source_kind": "google_slide_export",
                "file_id": file_id,
                "download_url": f"https://docs.google.com/presentation/d/{file_id}/export/pdf",
            }
        return {}

    if host == "docs.google.com" and len(path_parts) >= 3 and path_parts[1] == "d":
        kind = path_parts[0]
        file_id = path_parts[2]
        if kind in {"document", "spreadsheets", "presentation"} and file_id:
            return _build_doc_export(kind, file_id)

    if host == "drive.google.com":
        file_id = ""
        if len(path_parts) >= 3 and path_parts[0] == "file" and path_parts[1] == "d":
            file_id = path_parts[2]
        elif path_parts[:1] in (["open"], ["uc"]):
            file_id = (query.get("id") or [""])[0]
        elif not path_parts:
            file_id = (query.get("id") or [""])[0]
        if file_id:
            return {
                "source_kind": "google_drive_file",
                "file_id": file_id,
                "download_url": f"https://drive.google.com/uc?export=download&id={file_id}",
            }

    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _span_x0(span: Dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 1:
        return _safe_float(bbox[0])
    origin = span.get("origin")
    if isinstance(origin, (list, tuple)) and len(origin) >= 1:
        return _safe_float(origin[0])
    return 0.0


def _span_x1(span: Dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 3:
        return _safe_float(bbox[2], _span_x0(span))
    return _span_x0(span)


def _bbox_payload(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {
            "x0": _safe_float(value.get("x0")),
            "y0": _safe_float(value.get("y0")),
            "x1": _safe_float(value.get("x1")),
            "y1": _safe_float(value.get("y1")),
        }
    if isinstance(value, (list, tuple)):
        return {
            "x0": _safe_float(value[0]) if len(value) > 0 else 0.0,
            "y0": _safe_float(value[1]) if len(value) > 1 else 0.0,
            "x1": _safe_float(value[2]) if len(value) > 2 else 0.0,
            "y1": _safe_float(value[3]) if len(value) > 3 else 0.0,
        }
    return {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}


def _should_insert_span_space(previous: str, current: str, x_gap: float) -> bool:
    if not previous or not current:
        return False
    prev_ch = previous[-1]
    next_ch = current[0]
    if prev_ch.isspace() or next_ch.isspace():
        return False
    if next_ch in ".,;:!?)]}":
        return False
    if prev_ch in "([{/":
        return False
    if x_gap > 1.5:
        return True
    return prev_ch.isalnum() and next_ch.isalnum()


def _join_line_spans(spans: List[Dict[str, Any]]) -> str:
    ordered = sorted(spans, key=_span_x0)
    pieces: List[str] = []
    prev_text = ""
    prev_x1: Optional[float] = None
    for span in ordered:
        span_text = _sanitize_extracted_text(span.get("text"))
        if not span_text.strip():
            continue
        x0 = _span_x0(span)
        if pieces and prev_x1 is not None and _should_insert_span_space(prev_text, span_text, x0 - prev_x1):
            pieces.append(" ")
        pieces.append(span_text)
        prev_text = span_text
        prev_x1 = _span_x1(span)
    if not pieces:
        return ""
    return re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()


async def _fetch_text(url: str) -> str:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=30,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def _download_pdf(url: str, out_path: Path) -> None:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=60,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.content
        if not _looks_like_pdf_bytes(data):
            raise RuntimeError("Resolved source did not return a PDF. Ensure the document is shared/exportable as PDF.")
        out_path.write_bytes(data)


def _guess_title_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}
        title = (metadata.title or "").strip()
        if title:
            return title
        first = reader.pages[0].extract_text() or ""
        first = first.strip().split("\n", 1)[0][:160]
        return first or pdf_path.name
    except Exception:
        return pdf_path.name


async def resolve_any_to_pdf(input_str: str, output_dir: Optional[Path] = None) -> Tuple[str, Path]:
    """
    Resolve DOI, landing page URL, arXiv URL, or direct PDF URL into a local PDF.

    Returns:
        (title, local_pdf_path)
    """
    pdf_dir = (output_dir or _default_pdf_dir()).expanduser().resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)

    value = input_str.strip()
    if not value:
        raise ValueError("input_str cannot be empty")

    google_source = describe_google_drive_source(value)
    if google_source:
        pdf_url = google_source["download_url"]
        out = pdf_dir / _safe_filename(f"{google_source['file_id']}.pdf")
        await _download_pdf(pdf_url, out)
        return _guess_title_from_pdf(out), out

    # DOI shortcut
    if re.match(r"^10\.\d{4,9}/", value):
        landing_url = f"https://doi.org/{value}"
        html = await _fetch_text(landing_url)
    else:
        if value.lower().endswith(".pdf"):
            pdf_url = value
            out = pdf_dir / _safe_filename(pdf_url)
            await _download_pdf(pdf_url, out)
            return _guess_title_from_pdf(out), out
        html = await _fetch_text(value)

    soup = BeautifulSoup(html, "html.parser")
    pdf_url: Optional[str] = None
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and meta.get("content", "").strip():
        pdf_url = meta["content"].strip()

    if not pdf_url:
        link = soup.find("a", href=re.compile(r"\.pdf($|\?)", re.I))
        if link:
            href = str(link.get("href") or "").strip()
            if href.startswith("//"):
                href = "https:" + href
            pdf_url = href

    if not pdf_url:
        raise RuntimeError("Could not locate a PDF link on the source page.")

    out = pdf_dir / _safe_filename(pdf_url)
    await _download_pdf(pdf_url, out)
    return _guess_title_from_pdf(out), out


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Simple page-level text extraction using pypdf.
    """
    pages: List[Tuple[int, str]] = []
    reader = PdfReader(str(pdf_path))
    for idx, page in enumerate(reader.pages):
        text = _sanitize_extracted_text(page.extract_text())
        pages.append((idx + 1, text))
    return pages


def extract_text_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Block-level extraction using PyMuPDF with geometry + text style metadata.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    blocks: List[Dict[str, Any]] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Use MuPDF's sort order to stabilize block reading order across PDFs.
            page_dict = page.get_text("dict", sort=True)
            text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]

            block_idx = 0
            for block in text_blocks:
                lines = block.get("lines", [])
                text_lines: List[str] = []
                line_payloads: List[Dict[str, Any]] = []
                span_sizes: List[float] = []
                span_fonts: List[str] = []
                bold_spans = 0
                total_spans = 0

                for line in lines:
                    spans = line.get("spans", [])
                    line_spans: List[Dict[str, Any]] = []
                    for span in spans:
                        span_text = _sanitize_extracted_text(span.get("text"))
                        if not span_text.strip():
                            continue
                        size = span.get("size")
                        try:
                            span_sizes.append(float(size))
                        except (TypeError, ValueError):
                            pass
                        font_name = str(span.get("font") or "")
                        if font_name:
                            span_fonts.append(font_name)
                        total_spans += 1
                        if "bold" in font_name.lower():
                            bold_spans += 1
                        line_spans.append(
                            {
                                "text": span_text,
                                "bbox": _bbox_payload(span.get("bbox")),
                            }
                        )
                    line_text = _join_line_spans(spans)
                    if line_text:
                        text_lines.append(line_text)
                        line_payloads.append(
                            {
                                "text": line_text,
                                "bbox": _bbox_payload(line.get("bbox")),
                                "spans": line_spans,
                            }
                        )

                text = _sanitize_extracted_text("\n".join(text_lines).strip())
                if not text:
                    continue

                bbox = _bbox_payload(block.get("bbox", [0, 0, 0, 0]))
                first_line = text_lines[0].strip() if text_lines else text.splitlines()[0].strip()
                max_font = max(span_sizes) if span_sizes else 0.0
                avg_font = (sum(span_sizes) / len(span_sizes)) if span_sizes else 0.0
                min_font = min(span_sizes) if span_sizes else 0.0
                bold_ratio = (bold_spans / total_spans) if total_spans else 0.0

                blocks.append(
                    {
                        "page_no": page_num + 1,
                        "block_index": block_idx,
                        "text": text,
                        "bbox": bbox,
                        "metadata": {
                            "first_line": first_line,
                            "line_count": len(text_lines),
                            "char_count": len(text),
                            "max_font_size": round(max_font, 3),
                            "avg_font_size": round(avg_font, 3),
                            "min_font_size": round(min_font, 3),
                            "bold_ratio": round(bold_ratio, 3),
                            "font_names": sorted(set(span_fonts))[:6],
                            "lines": line_payloads,
                        },
                    }
                )
                block_idx += 1
    finally:
        doc.close()

    return blocks