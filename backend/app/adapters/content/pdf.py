import io
import os
import re
from collections import defaultdict
from urllib.parse import urlparse

import pdfplumber
import requests

from backend.app.domain.ports import ContentExtractor
from backend.app.domain.models import ExtractedContent
from backend.app.domain.errors import ExtractionError
from backend.app.util.text_processing import clean_document_text


def _debug_enabled() -> bool:
    val = os.getenv("APP_DEBUG", "")
    return val.lower() in ("1", "true", "yes", "on")


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[pdf_extractor] {msg}")


def _build_lines_from_chars(chars):
    if not chars:
        return []
    sorted_chars = sorted(chars, key=lambda c: (c.get("top", 0.0), c.get("x0", 0.0)))
    lines = []
    current = []
    current_top = None
    threshold = 3.0
    for ch in sorted_chars:
        top = float(ch.get("top", 0.0))
        if current and current_top is not None and abs(top - current_top) > threshold:
            lines.append(current)
            current = [ch]
            current_top = top
        else:
            current.append(ch)
            if current_top is None:
                current_top = top
    if current:
        lines.append(current)
    line_dicts = []
    for line_chars in lines:
        sorted_line_chars = sorted(line_chars, key=lambda c: c.get("x0", 0.0))
        widths = [float(c.get("x1", 0.0)) - float(c.get("x0", 0.0)) for c in sorted_line_chars]
        avg_width = sum(widths) / len(widths) if widths else 0.0
        parts = []
        prev = None
        for ch in sorted_line_chars:
            ch_text = ch.get("text", "")
            if prev is not None:
                prev_text = prev.get("text", "")
                if not prev_text.isspace() and not ch_text.isspace():
                    prev_x1 = float(prev.get("x1", prev.get("x0", 0.0)))
                    cur_x0 = float(ch.get("x0", prev_x1))
                    gap = cur_x0 - prev_x1
                    if avg_width > 0 and gap > avg_width * 0.5:
                        parts.append(" ")
            parts.append(ch_text)
            prev = ch
        text = "".join(parts)
        if not text.strip():
            continue
        sizes = [float(c.get("size", 0.0)) for c in sorted_line_chars]
        font_size = sum(sizes) / len(sizes) if sizes else 0.0
        tops = [float(c.get("top", 0.0)) for c in sorted_line_chars]
        bottoms = [float(c.get("bottom", 0.0)) for c in sorted_line_chars]
        font_bold = any("Bold" in str(c.get("fontname", "")) for c in sorted_line_chars)
        line_dicts.append(
            {
                "text": text,
                "font_size": font_size,
                "top": min(tops) if tops else 0.0,
                "bottom": max(bottoms) if bottoms else 0.0,
                "font_bold": font_bold,
            }
        )
    return line_dicts


def _find_header_footer_texts(pages_lines):
    header_positions = defaultdict(list)
    footer_positions = defaultdict(list)
    page_count = len(pages_lines)
    for page_data in pages_lines:
        page = page_data.get("lines") or []
        page_height = float(page_data.get("height") or 0.0)
        if not page or page_height <= 0.0:
            continue
        for line in page:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            norm = re.sub(r"\s+", " ", text).strip()
            top = float(line.get("top") or 0.0)
            bottom = float(line.get("bottom") or 0.0)
            rel_top = top / page_height
            rel_bottom = bottom / page_height
            if rel_top < 0.12:
                header_positions[norm].append(rel_top)
            if rel_bottom > 0.88:
                footer_positions[norm].append(rel_bottom)
    if page_count == 0:
        return set(), set()
    min_pages = max(2, int(page_count * 0.4))
    header_texts = {t for t, vals in header_positions.items() if len(vals) >= min_pages}
    footer_texts = {t for t, vals in footer_positions.items() if len(vals) >= min_pages}
    return header_texts, footer_texts


def _detect_sections_from_lines(pages_lines):
    header_texts, footer_texts = _find_header_footer_texts(pages_lines)
    sections = []
    for page_index, page_data in enumerate(pages_lines):
        page = page_data.get("lines") or []
        page_height = float(page_data.get("height") or 0.0)
        if not page:
            continue
        mean_font = sum(l["font_size"] for l in page) / len(page) if page else 0.0
        sorted_by_top = sorted(page, key=lambda l: l["top"])
        gaps = []
        for i in range(len(sorted_by_top) - 1):
            gap = sorted_by_top[i + 1]["top"] - sorted_by_top[i]["top"]
            if gap > 0:
                gaps.append(gap)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        line_to_gap = {}
        for i, line in enumerate(sorted_by_top):
            prev_gap = sorted_by_top[i]["top"] - sorted_by_top[i - 1]["top"] if i > 0 else None
            next_gap = (
                sorted_by_top[i + 1]["top"] - sorted_by_top[i]["top"]
                if i + 1 < len(sorted_by_top)
                else None
            )
            neighbor_gaps = [g for g in (prev_gap, next_gap) if g is not None and g > 0]
            nearest = min(neighbor_gaps) if neighbor_gaps else 0.0
            line_to_gap[id(line)] = nearest
        candidates = []
        for line in page:
            text = line["text"].strip()
            if not text:
                continue
            if len(text) > 200:
                continue
            tokens = text.split()
            if len(tokens) <= 1:
                continue
            if len(tokens) > 24:
                continue
            first_tok = tokens[0]
            second_tok = tokens[1] if len(tokens) > 1 else ""
            if first_tok and first_tok[0].isdigit() and second_tok and second_tok[0].islower():
                continue
            font_size = line["font_size"]
            top = line["top"]
            bottom = line["bottom"]
            norm = re.sub(r"\s+", " ", text).strip()
            rel_top = top / page_height if page_height > 0 else 0.0
            rel_bottom = bottom / page_height if page_height > 0 else 0.0
            is_header = norm in header_texts and rel_top < 0.2
            is_footer = norm in footer_texts and rel_bottom > 0.8
            if is_header or is_footer:
                continue
            near_top = page_height > 0 and top <= page_height * 0.25
            nearest_gap = line_to_gap.get(id(line), 0.0)
            isolated = avg_gap > 0 and nearest_gap > avg_gap * 1.5
            contains_sentence_break = bool(re.search(r"\.\s+[A-Z][a-z]", text))
            ends_with_sentence_punct = bool(re.search(r"[.!?]\s*$", text))
            numbered_match = re.match(r"^\s*(\d+(\.\d+)*\.?|[IVXLCDM]+\.?)\s+", text)
            numbered = numbered_match is not None
            heading_like = True
            if contains_sentence_break and len(tokens) >= 6:
                heading_like = False
            if ends_with_sentence_punct and not numbered and len(tokens) >= 8:
                heading_like = False
            if mean_font > 0:
                if not numbered:
                    if font_size < mean_font * 1.2 and not (isolated and font_size >= mean_font * 1.05):
                        heading_like = False
            if not heading_like:
                continue
            if not (near_top or isolated or numbered):
                continue
            all_caps = text.upper() == text and any(c.isalpha() for c in text)
            if numbered:
                numbering = numbered_match.group(1)
                if numbering:
                    level = numbering.count(".") + 1
                else:
                    level = 1
            else:
                level = 1
            score = font_size
            if near_top:
                score += font_size * 0.1
            if line["font_bold"]:
                score += font_size * 0.05
            if all_caps:
                score += font_size * 0.05
            if numbered:
                score += font_size * 0.05
            candidates.append(
                {
                    "title": text,
                    "level": level,
                    "font_size": font_size,
                    "page_index": page_index,
                    "score": score,
                }
            )
        if len(candidates) > 8:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            candidates = candidates[:8]
        if _debug_enabled():
            for cand in candidates:
                _debug(
                    f"section_candidate page={page_index} title={cand['title']!r} "
                    f"level={cand['level']} font={cand['font_size']:.2f} score={cand['score']:.2f}"
                )
        sections.extend(candidates)
    first_level_sections = [s for s in sections if s.get("level", 1) == 1]
    if not first_level_sections:
        best = None
        for page_index, page_data in enumerate(pages_lines[:2]):
            page = page_data.get("lines") or []
            page_height = float(page_data.get("height") or 0.0)
            if not page or page_height <= 0.0:
                continue
            mean_font = sum(l["font_size"] for l in page) / len(page) if page else 0.0
            sorted_by_top = sorted(page, key=lambda l: l["top"])
            gaps = []
            for i in range(len(sorted_by_top) - 1):
                gap = sorted_by_top[i + 1]["top"] - sorted_by_top[i]["top"]
                if gap > 0:
                    gaps.append(gap)
            avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
            line_to_gap = {}
            for i, line in enumerate(sorted_by_top):
                prev_gap = sorted_by_top[i]["top"] - sorted_by_top[i - 1]["top"] if i > 0 else None
                next_gap = (
                    sorted_by_top[i + 1]["top"] - sorted_by_top[i]["top"]
                    if i + 1 < len(sorted_by_top)
                    else None
                )
                neighbor_gaps = [g for g in (prev_gap, next_gap) if g is not None and g > 0]
                nearest = min(neighbor_gaps) if neighbor_gaps else 0.0
                line_to_gap[id(line)] = nearest
            for line in page:
                text = (line.get("text") or "").strip()
                if not text:
                    continue
                norm = re.sub(r"\s+", " ", text).strip()
                top = float(line.get("top") or 0.0)
                bottom = float(line.get("bottom") or 0.0)
                rel_top = top / page_height if page_height > 0.0 else 0.0
                rel_bottom = bottom / page_height if page_height > 0.0 else 0.0
                is_header = norm in header_texts and rel_top < 0.2
                is_footer = norm in footer_texts and rel_bottom > 0.8
                if is_header or is_footer:
                    continue
                tokens = text.split()
                if not tokens or len(tokens) > 24:
                    continue
                nearest_gap = line_to_gap.get(id(line), 0.0)
                isolated = avg_gap > 0 and nearest_gap > avg_gap * 1.5
                if not isolated:
                    continue
                font_size = float(line.get("font_size") or 0.0)
                if mean_font > 0 and font_size < mean_font * 1.2:
                    continue
                if not (page_index == 0 and rel_top <= 0.5):
                    continue
                contains_sentence_break = bool(re.search(r"\.\s+[A-Z][a-z]", text))
                ends_with_sentence_punct = bool(re.search(r"[.!?]\s*$", text))
                if contains_sentence_break:
                    continue
                if ends_with_sentence_punct and len(tokens) >= 8:
                    continue
                cand = {
                    "title": text,
                    "level": 1,
                    "font_size": font_size,
                    "page_index": page_index,
                    "score": font_size,
                }
                if best is None or font_size > best["font_size"]:
                    best = cand
        if best is not None:
            sections.append(best)
    return sections


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _collapse_alpha_spaces(text: str):
    chars = []
    index_map = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch.isspace():
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            if prev.isalpha() and nxt.isalpha():
                continue
        chars.append(ch)
        index_map.append(i)
    return "".join(chars), index_map


def _align_sections_to_text(text: str, candidates):
    if not text or not candidates:
        return []
    norm_text = _normalize_ws(text)
    norm_text_lower = norm_text.lower()
    sections = []
    cursor = 0
    collapsed_text_lower = None
    collapsed_map = None
    for cand in candidates:
        title = cand["title"]
        norm_title = _normalize_ws(title)
        if not norm_title:
            continue
        lower_title = norm_title.lower()
        idx = norm_text_lower.find(lower_title, cursor)
        store_title = norm_title
        if idx == -1:
            stripped = re.sub(r"^\s*(\d+(\.\d+)*\.?|[IVXLCDM]+\.?)\s+", "", norm_title, flags=re.IGNORECASE)
            stripped = _normalize_ws(stripped)
            if stripped and stripped.lower() != lower_title:
                idx = norm_text_lower.find(stripped.lower(), cursor)
        if idx == -1:
            compressed = re.sub(r"\s+", "", norm_title)
            compressed_lower = compressed.lower()
            if compressed_lower and compressed_lower != lower_title:
                alt_idx = norm_text_lower.find(compressed_lower, cursor)
                if alt_idx != -1:
                    idx = alt_idx
                    store_title = compressed
        if idx == -1:
            if collapsed_text_lower is None or collapsed_map is None:
                collapsed_text, collapsed_map = _collapse_alpha_spaces(norm_text)
                collapsed_text_lower = collapsed_text.lower()
            collapsed_cursor = 0
            while collapsed_cursor < len(collapsed_map) and collapsed_map[collapsed_cursor] < cursor:
                collapsed_cursor += 1
            alt_idx = collapsed_text_lower.find(lower_title, collapsed_cursor)
            if alt_idx != -1:
                idx = collapsed_map[alt_idx]
            else:
                compressed = re.sub(r"\s+", "", norm_title)
                compressed_lower = compressed.lower()
                if compressed_lower:
                    alt_idx = collapsed_text_lower.find(compressed_lower, collapsed_cursor)
                    if alt_idx != -1:
                        idx = collapsed_map[alt_idx]
                        store_title = compressed
        if idx == -1:
            if _debug_enabled():
                _debug(f"section_align_miss title={norm_title!r}")
            continue
        sections.append(
            {
                "title": store_title,
                "level": cand["level"],
                "start": idx,
                "end": None,
            }
        )
        cursor = idx + len(norm_title)
    if not sections:
        return []
    sections.sort(key=lambda s: s["start"])
    for i, s in enumerate(sections):
        if i + 1 < len(sections):
            s["end"] = sections[i + 1]["start"]
        else:
            s["end"] = None
    if _debug_enabled():
        for s in sections:
            _debug(
                f"section_final title={s['title']!r} level={s['level']} "
                f"start={s['start']} end={s['end']}"
            )
    return sections


def _guess_title_from_pages(pages_lines):
    if not pages_lines:
        return None
    first = pages_lines[0]
    lines = first.get("lines") or []
    if not lines:
        return None
    height = float(first.get("height") or 0.0)
    if height <= 0.0:
        height = max((l.get("bottom", 0.0) for l in lines), default=0.0)
    top_limit = height * 0.4 if height > 0.0 else 100.0
    candidates = []
    for line in lines:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        if len(text) < 3 or len(text) > 120:
            continue
        if float(line.get("top", 0.0)) > top_limit:
            continue
        lower = text.lower()
        if "http://" in lower or "https://" in lower:
            continue
        if re.match(r"^\d{4}\b", text.strip()):
            continue
        score = float(line.get("font_size", 0.0))
        candidates.append((score, text))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    best = re.sub(r"([0-9a-z])([A-Z])", r"\1 \2", best)
    return best


def _build_text_from_words(page) -> str:
    try:
        words = page.extract_words()
    except Exception:
        words = None
    if not words:
        return page.extract_text() or ""
    texts = [w.get("text", "") for w in words if w.get("text", "").strip()]
    if not texts:
        return page.extract_text() or ""
    avg_len = sum(len(t) for t in texts) / len(texts)
    max_len = max(len(t) for t in texts)
    long_tokens = [t for t in texts if len(t) >= 20]
    long_ratio = len(long_tokens) / len(texts) if texts else 0.0
    if avg_len > 12 or max_len > 30 or long_ratio > 0.2:
        return _build_text_from_chars(page)
    sorted_words = sorted(
        words,
        key=lambda w: (float(w.get("top", 0.0)), float(w.get("x0", 0.0))),
    )
    lines = []
    current = []
    current_top = None
    threshold = 3.0
    for w in sorted_words:
        top = float(w.get("top", 0.0))
        if current and current_top is not None and abs(top - current_top) > threshold:
            lines.append(current)
            current = [w]
            current_top = top
        else:
            current.append(w)
            if current_top is None:
                current_top = top
    if current:
        lines.append(current)
    line_texts = []
    for line_words in lines:
        sorted_line_words = sorted(line_words, key=lambda ww: ww.get("x0", 0.0))
        parts = [w.get("text", "") for w in sorted_line_words]
        line = " ".join(t for t in parts if t.strip())
        if line.strip():
            line_texts.append(line)
    return "\n".join(line_texts)


def _build_text_from_chars(page) -> str:
    chars = page.chars
    lines = _build_lines_from_chars(chars)
    if not lines:
        return page.extract_text() or ""
    sorted_by_top = sorted(lines, key=lambda l: l["top"])
    gaps = []
    for i in range(len(sorted_by_top) - 1):
        gap = sorted_by_top[i + 1]["top"] - sorted_by_top[i]["top"]
        if gap > 0:
            gaps.append(gap)
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    parts = []
    prev_line = None
    for line in sorted_by_top:
        text = line["text"].strip()
        if not text:
            prev_line = line
            continue
        if prev_line is not None:
            gap = line["top"] - prev_line["top"]
            if avg_gap > 0 and gap > avg_gap * 1.5:
                sep = "\n\n"
            else:
                sep = " "
            parts.append(sep)
        parts.append(text)
        prev_line = line
    return "".join(parts)


def _fix_pdf_urls(text: str) -> str:
    text = re.sub(r"\b(https?)\s*:\s*/\s*/\s*", lambda m: f"{m.group(1).lower()}://", text, flags=re.IGNORECASE)
    text = re.sub(
        r"([A-Za-z0-9_-])\s*\.\s*([A-Za-z]{2,})(\s*/)",
        r"\1.\2\3",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\bhttps?://[A-Za-z0-9_.\-/]+)\s*/\s*([A-Za-z0-9_.\-/]+)",
        r"\1/\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(https://github\.com/)([A-Za-z0-9 _-]+)\s*/\s*([A-Za-z0-9_.-]+)",
        lambda m: m.group(1) + re.sub(r'\s+', '', m.group(2)) + '/' + m.group(3),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(https?://)\s+",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    return text

class PDFExtractor(ContentExtractor):
    def extract(self, document_uri: str) -> ExtractedContent:
        try:
            parsed = urlparse(document_uri)
            if parsed.scheme in ("http", "https"):
                response = requests.get(document_uri, timeout=10)
                response.raise_for_status()
                data = response.content
            else:
                path = document_uri
                if parsed.scheme == "file":
                    path = parsed.path
                with open(path, "rb") as fh:
                    data = fh.read()

            f = io.BytesIO(data)

            with pdfplumber.open(f) as pdf:
                pages_lines = []
                page_texts = []
                for page in pdf.pages:
                    chars = page.chars
                    lines = _build_lines_from_chars(chars)
                    page_height = float(getattr(page, "height", 0.0) or 0.0)
                    page_line_infos = []
                    for line in lines:
                        text = line["text"].strip()
                        if not text:
                            continue
                        page_line_infos.append(
                            {
                                "text": text,
                                "font_size": line["font_size"],
                                "top": line["top"],
                                "bottom": line["bottom"],
                                "font_bold": line["font_bold"],
                            }
                        )
                    pages_lines.append(
                        {
                            "lines": page_line_infos,
                            "height": page_height,
                        }
                    )
                    page_text = _build_text_from_words(page)
                    if page_text:
                        page_texts.append(page_text.strip())
                if page_texts:
                    raw_text = "\n\n".join(page_texts)
                    if _debug_enabled():
                        _debug(f"raw_text_sample {raw_text[:200]!r}")
                    cleaned_text = clean_document_text(raw_text)
                    cleaned_text = _fix_pdf_urls(cleaned_text)
                    if _debug_enabled():
                        _debug(f"cleaned_text_sample {cleaned_text[:200]!r}")
                    normalized_text = _normalize_ws(cleaned_text or raw_text)
                else:
                    cleaned_text = ""
                    normalized_text = ""
                if not (cleaned_text or normalized_text):
                    fallback_pages = []
                    for page in pdf.pages:
                        t = page.extract_text() or ""
                        t = t.strip()
                        if t:
                            fallback_pages.append(t)
                    if fallback_pages:
                        raw_text = "\n\n".join(fallback_pages)
                        cleaned_text = clean_document_text(raw_text)
                        normalized_text = _normalize_ws(cleaned_text or raw_text)
                if not (cleaned_text or normalized_text):
                    raw_str = data.decode("latin-1", errors="ignore")
                    matches = re.findall(r"\(([^()]*)\)\s*Tj", raw_str)
                    if matches:
                        raw_text = "\n".join(matches)
                        cleaned_text = clean_document_text(raw_text)
                        normalized_text = _normalize_ws(cleaned_text or raw_text)
                if normalized_text:
                    candidates = _detect_sections_from_lines(pages_lines)
                    sections = _align_sections_to_text(normalized_text, candidates)
                else:
                    sections = []
                page_count = len(pdf.pages)
                if page_count == 0:
                    counts = re.findall(r"/Count\s+(\d+)", data.decode("latin-1", errors="ignore"))
                    if counts:
                        try:
                            page_count = max(int(c) for c in counts)
                        except Exception:
                            page_count = 0
                info = pdf.metadata or {}
                title = info.get("Title") if info else None
                if title:
                    title = title.strip() or None
                if not title:
                    title = _guess_title_from_pages(pages_lines)
                extra = {"page_count": page_count}
                if sections:
                    extra["sections"] = sections
                return ExtractedContent(
                    text=cleaned_text or normalized_text or "",
                    title=title,
                    mime_type="application/pdf",
                    extra=extra,
                )

        except Exception as e:
            raise ExtractionError(f"Failed to extract PDF content from {document_uri}: {str(e)}")
