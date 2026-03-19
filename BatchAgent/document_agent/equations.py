from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pymupdf

logger = logging.getLogger(__name__)

_EQUATION_NUMBER_RE = re.compile(r"\(\s*(\d+[A-Za-z]?)\s*\)\s*$")
_MATH_SYMBOL_RE = re.compile(r"[=+\-*/^_<>≤≥≈≠∈∉∀∃∑∫√πλμσθΔΩαβγ]")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_CAPTION_NOISE_RE = re.compile(r"^\s*(figure|fig\.?|table|tab\.?|algorithm)\s*\d+\b", re.IGNORECASE)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox_from_tuple(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return None
    return {
        "x0": _safe_float(value[0]),
        "y0": _safe_float(value[1]),
        "x1": _safe_float(value[2]),
        "y1": _safe_float(value[3]),
    }


def _bbox_from_payload(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    keys = ("x0", "y0", "x1", "y1")
    if not all(k in value for k in keys):
        return None
    bbox = {key: _safe_float(value.get(key)) for key in keys}
    if bbox["x1"] <= bbox["x0"] or bbox["y1"] <= bbox["y0"]:
        return None
    return bbox


def _rect_area(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox["x1"]) - float(bbox["x0"])) * max(0.0, float(bbox["y1"]) - float(bbox["y0"]))


def _rect_overlap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(float(a["x0"]), float(b["x0"]))
    iy0 = max(float(a["y0"]), float(b["y0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    iy1 = min(float(a["y1"]), float(b["y1"]))
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _bbox_union(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not a:
        return b
    if not b:
        return a
    return {
        "x0": min(float(a["x0"]), float(b["x0"])),
        "y0": min(float(a["y0"]), float(b["y0"])),
        "x1": max(float(a["x1"]), float(b["x1"])),
        "y1": max(float(a["y1"]), float(b["y1"])),
    }


def _center_x(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return (float(bbox["x0"]) + float(bbox["x1"])) * 0.5


def _center_y(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return (float(bbox["y0"]) + float(bbox["y1"])) * 0.5


def _bbox_width(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox["x1"]) - float(bbox["x0"]))


def _vertical_gap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return float("inf")
    if float(a["y1"]) < float(b["y0"]):
        return float(b["y0"]) - float(a["y1"])
    if float(b["y1"]) < float(a["y0"]):
        return float(a["y0"]) - float(b["y1"])
    return 0.0


def _x_overlap_ratio(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(float(a["x0"]), float(b["x0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    overlap = max(0.0, ix1 - ix0)
    base = max(1e-6, min(_bbox_width(a), _bbox_width(b)))
    return overlap / base


def _default_equation_dir() -> Path:
    return (Path.cwd() / ".ia_phase1_data" / "equations").expanduser().resolve()


def _equation_root() -> Path:
    configured = os.getenv("EQUATION_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _default_equation_dir()


def _paper_equation_dir(paper_id: int) -> Path:
    return _equation_root() / str(int(paper_id))


def _manifest_path(paper_id: int) -> Path:
    return _paper_equation_dir(paper_id) / "manifest.json"


def _equation_enabled() -> bool:
    raw = os.getenv("EQUATION_EXTRACTION_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


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


def _join_spans(spans: List[Dict[str, Any]]) -> str:
    ordered = sorted(spans, key=_span_x0)
    pieces: List[str] = []
    prev_text = ""
    prev_x1: Optional[float] = None
    for span in ordered:
        text = str(span.get("text") or "").replace("\x00", "")
        if not text.strip():
            continue
        x0 = _span_x0(span)
        if pieces and prev_x1 is not None and _should_insert_span_space(prev_text, text, x0 - prev_x1):
            pieces.append(" ")
        pieces.append(text)
        prev_text = text
        prev_x1 = _span_x1(span)
    return re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()


def _extract_block_lines(block: Dict[str, Any]) -> List[str]:
    lines_out: List[str] = []
    for line in block.get("lines", []):
        spans = line.get("spans")
        spans_list = spans if isinstance(spans, list) else []
        line_text = _join_spans(spans_list)
        if line_text:
            lines_out.append(line_text)
    return lines_out


def _extract_equation_number(lines: List[str]) -> Optional[str]:
    for line in reversed(lines):
        match = _EQUATION_NUMBER_RE.search(line.strip())
        if match:
            return str(match.group(1)).strip()
    return None


def _looks_like_equation_candidate(
    *,
    text_lines: List[str],
    bbox: Dict[str, float],
    page_width: float,
) -> Tuple[bool, float, List[str], Optional[str]]:
    compact = " ".join(text_lines).strip()
    if not compact:
        return False, 0.0, [], None
    if _CAPTION_NOISE_RE.match(compact):
        return False, 0.0, ["caption_noise"], None

    min_chars = max(4, _safe_int(os.getenv("EQUATION_MIN_CHARS", "8"), 8))
    if len(compact) < min_chars:
        return False, 0.0, ["too_short"], None

    equation_number = _extract_equation_number(text_lines)
    symbol_count = len(_MATH_SYMBOL_RE.findall(compact))
    has_equals = "=" in compact
    digit_count = sum(1 for ch in compact if ch.isdigit())
    word_count = len(_WORD_RE.findall(compact))
    line_count = len(text_lines)
    width_ratio = _bbox_width(bbox) / max(1.0, page_width)
    centered = abs(_center_x(bbox) - (page_width * 0.5)) / max(1.0, page_width) <= 0.20
    sentence_like = compact.count(". ") + compact.count(", ")

    score = 0.0
    flags: List[str] = []

    if equation_number:
        score += 3.0
        flags.append("equation_number")
    if has_equals:
        score += 2.0
        flags.append("has_equals")
    if symbol_count >= 2:
        score += 1.2
        flags.append("math_symbols")
    if digit_count >= 1:
        score += 0.8
        flags.append("has_digits")
    if 2 <= line_count <= 8:
        score += 1.0
        flags.append("multi_line")
    if centered:
        score += 0.7
        flags.append("centered")
    if width_ratio <= 0.9:
        score += 0.5
        flags.append("narrow")
    if word_count <= 24:
        score += 0.8
        flags.append("limited_prose")

    if word_count >= max(30, _safe_int(os.getenv("EQUATION_MAX_WORDS", "36"), 36)):
        score -= 2.5
        flags.append("prose_heavy")
    if sentence_like >= 4 and not equation_number:
        score -= 1.0
        flags.append("sentence_heavy")
    if symbol_count == 0 and not equation_number:
        score -= 2.0
        flags.append("no_math_symbols")
    if not has_equals and not equation_number and symbol_count < 3:
        score -= 1.5
        flags.append("weak_math_signal")

    min_score = _safe_float(os.getenv("EQUATION_DETECTION_MIN_SCORE", "5.0"), 5.0)
    return score >= min_score, round(score, 3), flags, equation_number


def _is_equation_number_only(candidate: Dict[str, Any]) -> bool:
    text = str(candidate.get("text") or "").strip()
    if not text:
        return False
    normalized = " ".join(text.split())
    if _EQUATION_NUMBER_RE.match(normalized):
        return True
    return len(normalized) <= 4 and normalized.startswith("(") and normalized.endswith(")")


def _merge_equation_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(candidates) <= 1:
        return candidates

    gap_limit = max(0.0, _safe_float(os.getenv("EQUATION_MERGE_VERTICAL_GAP_PT", "18"), 18.0))
    overlap_min = min(1.0, max(0.0, _safe_float(os.getenv("EQUATION_MERGE_X_OVERLAP_RATIO", "0.35"), 0.35)))

    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item.get("page_no") or 0),
            float(item["bbox"]["y0"]) if item.get("bbox") else 0.0,
            float(item["bbox"]["x0"]) if item.get("bbox") else 0.0,
        ),
    )
    merged: List[Dict[str, Any]] = []
    for candidate in ordered:
        if not merged:
            merged.append(candidate)
            continue

        prev = merged[-1]
        prev_bbox = _bbox_from_payload(prev.get("bbox"))
        cur_bbox = _bbox_from_payload(candidate.get("bbox"))
        if int(prev.get("page_no") or 0) != int(candidate.get("page_no") or 0):
            merged.append(candidate)
            continue

        can_merge = False
        if prev_bbox and cur_bbox:
            y_gap = float(cur_bbox["y0"]) - float(prev_bbox["y1"])
            x_overlap = _x_overlap_ratio(prev_bbox, cur_bbox)
            number_tail = _is_equation_number_only(candidate)
            if number_tail and y_gap <= 28.0:
                can_merge = True
            elif y_gap >= -2.0 and y_gap <= gap_limit and x_overlap >= overlap_min:
                can_merge = True

        if not can_merge:
            merged.append(candidate)
            continue

        prev_lines = prev.get("lines")
        if not isinstance(prev_lines, list):
            prev_lines = []
        cur_lines = candidate.get("lines")
        if not isinstance(cur_lines, list):
            cur_lines = []
        merged_lines = [str(line) for line in prev_lines + cur_lines if str(line).strip()]
        prev["lines"] = merged_lines
        prev["text"] = "\n".join(merged_lines).strip()
        prev["bbox"] = _bbox_union(prev_bbox, cur_bbox)
        prev["score"] = max(_safe_float(prev.get("score")), _safe_float(candidate.get("score")))

        prev_flags = [str(flag) for flag in (prev.get("flags") if isinstance(prev.get("flags"), list) else [])]
        cur_flags = [str(flag) for flag in (candidate.get("flags") if isinstance(candidate.get("flags"), list) else [])]
        for flag in cur_flags:
            if flag not in prev_flags:
                prev_flags.append(flag)
        prev["flags"] = prev_flags
        prev["equation_number"] = str(
            prev.get("equation_number")
            or candidate.get("equation_number")
            or _extract_equation_number(merged_lines)
            or ""
        ).strip() or None

    return merged


def _group_blocks_by_page(blocks: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        page_no = _safe_int(block.get("page_no"), 0)
        if page_no <= 0:
            continue
        by_page.setdefault(page_no, []).append(block)
    for page_no in by_page:
        by_page[page_no].sort(key=lambda item: _safe_int(item.get("block_index"), 0))
    return by_page


def _infer_section_for_equation(
    page_blocks: Iterable[Dict[str, Any]],
    equation_bbox: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    eq_area = max(_rect_area(equation_bbox), 1e-6)

    for block in page_blocks:
        if not isinstance(block, dict):
            continue
        block_meta = block.get("metadata")
        block_meta = block_meta if isinstance(block_meta, dict) else {}
        canonical = str(
            block_meta.get("section_canonical")
            or block_meta.get("section_primary")
            or ""
        ).strip()
        if not canonical:
            canonical = "other"
        title = str(block_meta.get("section_title") or canonical.replace("_", " ").title()).strip()
        source = str(block_meta.get("section_source") or "fallback").strip() or "fallback"
        confidence = block_meta.get("section_confidence")
        confidence_val = _safe_float(confidence, 0.35)
        block_bbox = _bbox_from_payload(block.get("bbox"))

        score = 0.0
        if equation_bbox and block_bbox:
            overlap = _rect_overlap(equation_bbox, block_bbox)
            block_area = max(_rect_area(block_bbox), 1e-6)
            if overlap > 0:
                score = (overlap / eq_area) * 0.75 + (overlap / block_area) * 0.25 + 0.5
            else:
                distance = _vertical_gap(equation_bbox, block_bbox)
                score = 1.0 / (1.0 + distance)
                if float(block_bbox["y1"]) <= float(equation_bbox["y0"]) + 2.0:
                    score += 0.08
        elif equation_bbox:
            score = 0.1
        else:
            score = 0.05

        if score > best_score:
            best_score = score
            best = {
                "section_canonical": canonical,
                "section_title": title or canonical.replace("_", " ").title(),
                "section_source": source,
                "section_confidence": round(confidence_val, 3),
            }

    if best:
        return best
    return {
        "section_canonical": "other",
        "section_title": "Other",
        "section_source": "fallback",
        "section_confidence": 0.35,
    }


def _render_equation_image(
    page: Any,
    bbox: Optional[Dict[str, float]],
    out_path: Path,
) -> bool:
    if not bbox:
        return False
    margin = max(0.0, _safe_float(os.getenv("EQUATION_CLIP_MARGIN_PT", "6"), 6.0))
    scale = max(1.0, _safe_float(os.getenv("EQUATION_RENDER_SCALE", "2.2"), 2.2))
    min_side_px = max(12, _safe_int(os.getenv("EQUATION_MIN_SIDE_PX", "20"), 20))
    min_area_px = max(200, _safe_int(os.getenv("EQUATION_MIN_PIXEL_AREA", "1200"), 1200))

    page_rect = page.rect
    clip_rect = pymupdf.Rect(
        max(page_rect.x0, float(bbox["x0"]) - margin),
        max(page_rect.y0, float(bbox["y0"]) - margin),
        min(page_rect.x1, float(bbox["x1"]) + margin),
        min(page_rect.y1, float(bbox["y1"]) + margin),
    )
    if clip_rect.width <= 1 or clip_rect.height <= 1:
        return False

    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip_rect, alpha=False)
    if pix.width < min_side_px or pix.height < min_side_px:
        return False
    if pix.width * pix.height < min_area_px:
        return False
    pix.save(str(out_path))
    return True


def extract_and_store_paper_equations(
    pdf_path: Path | str,
    paper_id: int,
    blocks: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect display equations in a PDF, map them to sections, and write manifest + crops.
    """
    if not _equation_enabled():
        return {
            "paper_id": int(paper_id),
            "num_equations": 0,
            "manifest_path": None,
            "equations": [],
        }

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    output_dir = _paper_equation_dir(paper_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("equation_*"):
        try:
            stale.unlink()
        except Exception:
            logger.warning("Failed to remove stale equation artifact %s", stale)
    manifest = _manifest_path(paper_id)
    if manifest.exists():
        try:
            manifest.unlink()
        except Exception:
            logger.warning("Failed to remove stale equation manifest %s", manifest)

    page_blocks = _group_blocks_by_page(blocks)
    equation_records: List[Dict[str, Any]] = []

    with pymupdf.open(str(path)) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_no = page_index + 1
            page_dict = page.get_text("dict")
            raw_text_blocks = [item for item in page_dict.get("blocks", []) if item.get("type") == 0]
            page_width = max(1.0, float(page.rect.width))

            candidates: List[Dict[str, Any]] = []
            for raw_block in raw_text_blocks:
                bbox = _bbox_from_tuple(raw_block.get("bbox"))
                if not bbox:
                    continue
                text_lines = _extract_block_lines(raw_block)
                if not text_lines:
                    continue
                ok, score, flags, equation_number = _looks_like_equation_candidate(
                    text_lines=text_lines,
                    bbox=bbox,
                    page_width=page_width,
                )
                if not ok:
                    continue
                candidates.append(
                    {
                        "page_no": page_no,
                        "bbox": bbox,
                        "lines": text_lines,
                        "text": "\n".join(text_lines).strip(),
                        "score": score,
                        "flags": flags,
                        "equation_number": equation_number,
                    }
                )

            for candidate in _merge_equation_candidates(candidates):
                equation_id = len(equation_records) + 1
                file_name = f"equation_{equation_id:04d}.png"
                image_path = output_dir / file_name
                image_saved = _render_equation_image(page, _bbox_from_payload(candidate.get("bbox")), image_path)
                section = _infer_section_for_equation(
                    page_blocks.get(page_no, []),
                    _bbox_from_payload(candidate.get("bbox")),
                )

                record = {
                    "id": equation_id,
                    "page_no": page_no,
                    "equation_number": candidate.get("equation_number"),
                    "text": str(candidate.get("text") or "").strip(),
                    "line_count": len(candidate.get("lines") or []),
                    "char_count": len(str(candidate.get("text") or "")),
                    "bbox": candidate.get("bbox"),
                    "detection_score": _safe_float(candidate.get("score")),
                    "detection_flags": candidate.get("flags") if isinstance(candidate.get("flags"), list) else [],
                    "section_canonical": section["section_canonical"],
                    "section_title": section["section_title"],
                    "section_source": section["section_source"],
                    "section_confidence": section["section_confidence"],
                    "file_name": file_name if image_saved else None,
                    "image_path": str(image_path) if image_saved else None,
                    "url": f"/api/papers/{paper_id}/equations/{file_name}" if image_saved else None,
                    "json_file": f"equation_{equation_id:04d}.json",
                }

                per_equation_path = output_dir / record["json_file"]
                with per_equation_path.open("w", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                equation_records.append(record)

    payload = {
        "paper_id": int(paper_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_equations": len(equation_records),
        "equations": equation_records,
    }
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return {
        "paper_id": int(paper_id),
        "num_equations": len(equation_records),
        "manifest_path": str(manifest),
        "equations": equation_records,
    }


def _next_indices_by_page(text_blocks: Iterable[Dict[str, Any]]) -> Dict[int, int]:
    max_by_page: Dict[int, int] = {}
    for block in text_blocks:
        if not isinstance(block, dict):
            continue
        page_no = _safe_int(block.get("page_no"), 0)
        if page_no <= 0:
            continue
        idx = _safe_int(block.get("block_index"), 0)
        max_by_page[page_no] = max(max_by_page.get(page_no, -1), idx)
    return {page_no: max_idx + 2000 for page_no, max_idx in max_by_page.items()}


def _nearest_context_snippet(
    *,
    page_no: int,
    equation_bbox: Optional[Dict[str, float]],
    text_blocks: Iterable[Dict[str, Any]],
    max_chars: int = 360,
) -> str:
    best_text = ""
    best_score = -1.0
    for block in text_blocks:
        if not isinstance(block, dict):
            continue
        if _safe_int(block.get("page_no"), 0) != page_no:
            continue
        text = str(block.get("text") or "").strip().replace("\x00", "")
        if not text:
            continue
        block_bbox = _bbox_from_payload(block.get("bbox"))
        score = 0.0
        if equation_bbox and block_bbox:
            overlap = _rect_overlap(equation_bbox, block_bbox)
            if overlap > 0:
                score = 3.0 + overlap / max(_rect_area(equation_bbox), 1.0)
            else:
                distance = _vertical_gap(equation_bbox, block_bbox)
                score = 1.0 / (1.0 + distance)
                if float(block_bbox["y1"]) <= float(equation_bbox["y0"]) + 1.5:
                    score += 0.06
        else:
            score = 0.01
        if score > best_score:
            best_score = score
            best_text = " ".join(text.split())
    return best_text[:max_chars]


def equation_records_to_chunks(
    equations: Iterable[Dict[str, Any]],
    text_blocks: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert equation records to chunk records for vector indexing.
    """
    records = [item for item in equations if isinstance(item, dict)]
    if not records:
        return []

    max_chars = max(300, _safe_int(os.getenv("EQUATION_CHUNK_MAX_CHARS", "1600"), 1600))
    next_by_page = _next_indices_by_page(text_blocks)
    chunks: List[Dict[str, Any]] = []

    text_blocks_list = [item for item in text_blocks if isinstance(item, dict)]
    for equation in records:
        page_no = _safe_int(equation.get("page_no"), 1)
        equation_id = _safe_int(equation.get("id"), 0)
        equation_number = str(equation.get("equation_number") or "").strip() or None
        equation_text = str(equation.get("text") or "").strip()
        if not equation_text:
            equation_text = "(Equation image extracted; exact symbol-level text unavailable.)"

        section_canonical = str(equation.get("section_canonical") or "other").strip() or "other"
        section_title = str(equation.get("section_title") or "Document Body").strip() or "Document Body"
        section_source = str(equation.get("section_source") or "fallback").strip() or "fallback"
        section_confidence = round(_safe_float(equation.get("section_confidence"), 0.35), 3)
        equation_bbox = _bbox_from_payload(equation.get("bbox"))
        context_snippet = _nearest_context_snippet(
            page_no=page_no,
            equation_bbox=equation_bbox,
            text_blocks=text_blocks_list,
            max_chars=300,
        )

        heading = f"Equation {equation_number}" if equation_number else f"Equation {equation_id}"
        parts = [heading, f"Section: {section_title}", equation_text]
        if context_snippet:
            parts.append(f"Nearby context: {context_snippet}")
        chunk_text = "\n\n".join(part for part in parts if part).strip()
        if len(chunk_text) > max_chars:
            chunk_text = chunk_text[:max_chars].rstrip()

        if page_no not in next_by_page:
            next_by_page[page_no] = 2000
        block_index = next_by_page[page_no]
        next_by_page[page_no] = block_index + 1

        source_block = {
            "page_no": page_no,
            "block_index": block_index,
            "text": equation_text,
            "bbox": equation_bbox,
            "metadata": {
                "section_title": section_title,
                "section_canonical": section_canonical,
                "section_source": section_source,
                "section_confidence": section_confidence,
            },
        }
        metadata = {
            "chunk_type": "equation",
            "content_type": "equation",
            "equation_id": equation_id,
            "equation_number": equation_number,
            "equation_page_no": page_no,
            "equation_file_name": equation.get("file_name"),
            "equation_url": equation.get("url"),
            "equation_detection_score": _safe_float(equation.get("detection_score"), 0.0),
            "equation_detection_flags": equation.get("detection_flags") if isinstance(equation.get("detection_flags"), list) else [],
            "section_primary": section_canonical,
            "section_all": [section_canonical],
            "section_titles": [section_title],
            "section_source": section_source,
            "section_confidence": section_confidence,
            "spans_multiple_sections": False,
            "blocks": [source_block],
        }

        chunks.append(
            {
                "text": chunk_text,
                "page_no": page_no,
                "block_index": block_index,
                "bbox": equation_bbox,
                "metadata": metadata,
            }
        )

    return chunks


def load_paper_equation_manifest(paper_id: int) -> Dict[str, Any]:
    path = _manifest_path(paper_id)
    if not path.exists():
        return {"paper_id": int(paper_id), "num_equations": 0, "equations": []}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read equation manifest for paper %s: %s", paper_id, exc)
        return {"paper_id": int(paper_id), "num_equations": 0, "equations": []}
    if not isinstance(payload, dict):
        return {"paper_id": int(paper_id), "num_equations": 0, "equations": []}
    equations = payload.get("equations")
    if not isinstance(equations, list):
        equations = []
    payload["equations"] = equations
    payload["num_equations"] = _safe_int(payload.get("num_equations"), len(equations))
    payload["paper_id"] = _safe_int(payload.get("paper_id"), int(paper_id))
    return payload


def resolve_equation_file(paper_id: int, file_name: str) -> Path:
    candidate = str(file_name or "").strip()
    if not candidate:
        raise ValueError("Equation file name is required.")
    if "/" in candidate or "\\" in candidate or ".." in candidate:
        raise ValueError("Invalid equation file name.")
    return _paper_equation_dir(paper_id) / candidate
