from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pymupdf

logger = logging.getLogger(__name__)

_TABLE_CAPTION_RE = re.compile(r"^\s*(table|tab\.)\s*\d+\b", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.)\s*\d+\b", re.IGNORECASE)
_BASE64_LIKE_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")


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


def _bbox_from_payload(payload: Any) -> Optional[Dict[str, float]]:
    if not isinstance(payload, dict):
        return None
    x0 = _safe_float(payload.get("x0"), float("nan"))
    y0 = _safe_float(payload.get("y0"), float("nan"))
    x1 = _safe_float(payload.get("x1"), float("nan"))
    y1 = _safe_float(payload.get("y1"), float("nan"))
    if any(v != v for v in (x0, y0, x1, y1)):
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _bbox_from_tuple(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return None
    return {
        "x0": _safe_float(value[0]),
        "y0": _safe_float(value[1]),
        "x1": _safe_float(value[2]),
        "y1": _safe_float(value[3]),
    }


def _rect_area(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    width = max(0.0, bbox["x1"] - bbox["x0"])
    height = max(0.0, bbox["y1"] - bbox["y0"])
    return width * height


def _rect_overlap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(a["x0"], b["x0"])
    iy0 = max(a["y0"], b["y0"])
    ix1 = min(a["x1"], b["x1"])
    iy1 = min(a["y1"], b["y1"])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _rect_iou(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    overlap = _rect_overlap(a, b)
    if overlap <= 0:
        return 0.0
    union = _rect_area(a) + _rect_area(b) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def _vertical_gap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return float("inf")
    if float(a["y1"]) < float(b["y0"]):
        return float(b["y0"]) - float(a["y1"])
    if float(b["y1"]) < float(a["y0"]):
        return float(a["y0"]) - float(b["y1"])
    return 0.0


def _center_y(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return (bbox["y0"] + bbox["y1"]) * 0.5


def _bbox_width(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox["x1"]) - float(bbox["x0"]))


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


def _x_overlap_ratio(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(float(a["x0"]), float(b["x0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    overlap = max(0.0, ix1 - ix0)
    base = max(1e-6, min(_bbox_width(a), _bbox_width(b)))
    return overlap / base


def _merge_table_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(candidates) <= 1:
        return candidates

    gap_limit = max(0.0, _safe_float(os.getenv("TABLE_MERGE_VERTICAL_GAP_PT", "14"), 14.0))
    overlap_min = min(1.0, max(0.0, _safe_float(os.getenv("TABLE_MERGE_X_OVERLAP_RATIO", "0.9"), 0.9)))

    ordered = sorted(candidates, key=lambda item: float(item["bbox"]["y0"]) if item.get("bbox") else 0.0)
    merged: List[Dict[str, Any]] = []
    for candidate in ordered:
        if not merged:
            merged.append(candidate)
            continue

        prev = merged[-1]
        prev_bbox = prev.get("bbox")
        cur_bbox = candidate.get("bbox")
        prev_cols = int(prev.get("n_cols") or 0)
        cur_cols = int(candidate.get("n_cols") or 0)

        can_merge = False
        if prev_bbox and cur_bbox and prev_cols > 0 and cur_cols > 0:
            y_gap = float(cur_bbox["y0"]) - float(prev_bbox["y1"])
            x_overlap = _x_overlap_ratio(prev_bbox, cur_bbox)
            rows_small = int(prev.get("raw_row_count") or 0) <= 2 or int(candidate.get("raw_row_count") or 0) <= 2
            if y_gap >= -2.0 and y_gap <= gap_limit and x_overlap >= overlap_min and prev_cols == cur_cols and rows_small:
                can_merge = True

        if can_merge:
            prev["matrix"].extend(candidate.get("matrix") or [])
            prev["bbox"] = _bbox_union(prev_bbox, cur_bbox)
            prev["raw_row_count"] = int(prev.get("raw_row_count") or 0) + int(candidate.get("raw_row_count") or 0)
            prev["table_obj"] = None
            prev["merged_fragments"] = int(prev.get("merged_fragments") or 1) + int(candidate.get("merged_fragments") or 1)
        else:
            merged.append(candidate)
    return merged


def _bbox_matches_any_iou(
    bbox: Optional[Dict[str, float]],
    existing: Iterable[Optional[Dict[str, float]]],
    threshold: float,
) -> bool:
    if not bbox:
        return False
    for other in existing:
        if _rect_iou(bbox, other) >= threshold:
            return True
    return False


def _extract_page_text_blocks(page: Any) -> List[Dict[str, Any]]:
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return []
    items: List[Dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = _bbox_from_tuple(block.get("bbox"))
        if not bbox:
            continue
        text = _extract_block_text(block)
        if not text:
            continue
        items.append({"bbox": bbox, "text": text})
    items.sort(key=lambda item: (float(item["bbox"]["y0"]), float(item["bbox"]["x0"])))
    return items


def _collect_caption_blocks(page: Any, pattern: re.Pattern[str]) -> List[Dict[str, Any]]:
    return [item for item in _extract_page_text_blocks(page) if pattern.search(item.get("text") or "")]


def _bbox_to_rect_tuple(bbox: Optional[Dict[str, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox:
        return None
    return (
        float(bbox["x0"]),
        float(bbox["y0"]),
        float(bbox["x1"]),
        float(bbox["y1"]),
    )


def _build_candidates_from_table_objects(
    table_objects: Iterable[Any],
    *,
    min_area: float,
    min_cols: int,
    detection_strategy: str,
    seed_caption: Optional[str] = None,
    seed_caption_id: Optional[int] = None,
    seed_caption_bbox: Optional[Dict[str, float]] = None,
    clip_bbox: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for table_obj in table_objects:
        bbox = _bbox_from_tuple(getattr(table_obj, "bbox", None))
        if _rect_area(bbox) < min_area:
            continue
        try:
            raw_rows = table_obj.extract()
        except Exception:
            raw_rows = []
        matrix = _normalize_matrix(raw_rows)
        if not matrix:
            continue
        n_cols_raw = len(matrix[0]) if matrix else 0
        if n_cols_raw < min_cols:
            continue
        candidates.append(
            {
                "bbox": bbox,
                "matrix": matrix,
                "table_obj": table_obj,
                "n_cols": n_cols_raw,
                "raw_row_count": len(matrix),
                "merged_fragments": 1,
                "detection_strategy": detection_strategy,
                "seed_caption": seed_caption,
                "seed_caption_id": seed_caption_id,
                "seed_caption_bbox": seed_caption_bbox,
                "clip_bbox": clip_bbox,
            }
        )
    return candidates


def _page_bbox(page: Any) -> Dict[str, float]:
    rect = page.rect
    return {
        "x0": float(rect.x0),
        "y0": float(rect.y0),
        "x1": float(rect.x1),
        "y1": float(rect.y1),
    }


def _passes_text_fallback_constraints(
    table_bbox: Optional[Dict[str, float]],
    page_bbox: Optional[Dict[str, float]],
    caption_bbox: Optional[Dict[str, float]],
) -> bool:
    if not table_bbox or not page_bbox or not caption_bbox:
        return False

    max_caption_gap = max(0.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_CAPTION_GAP_PT", "110"), 110.0))
    if _vertical_gap(table_bbox, caption_bbox) > max_caption_gap:
        return False

    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MIN_X_OVERLAP_RATIO", "0.15"), 0.15)))
    if _x_overlap_ratio(table_bbox, caption_bbox) < min_x_overlap:
        return False

    page_area = max(1.0, _rect_area(page_bbox))
    max_page_area_ratio = min(1.0, max(0.01, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MAX_PAGE_AREA_RATIO", "0.65"), 0.65)))
    if (_rect_area(table_bbox) / page_area) > max_page_area_ratio:
        return False

    page_width = max(1e-6, _bbox_width(page_bbox))
    min_width_ratio = min(1.0, max(0.01, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MIN_WIDTH_RATIO", "0.20"), 0.20)))
    if (_bbox_width(table_bbox) / page_width) < min_width_ratio:
        return False

    return True


def _score_text_fallback_candidate(
    candidate: Dict[str, Any],
    *,
    caption_bbox: Dict[str, float],
    page_bbox: Dict[str, float],
) -> float:
    bbox = candidate.get("bbox")
    if not isinstance(bbox, dict):
        return -10**9
    gap = _vertical_gap(bbox, caption_bbox)
    x_overlap = _x_overlap_ratio(bbox, caption_bbox)
    page_area = max(1.0, _rect_area(page_bbox))
    area_ratio = _rect_area(bbox) / page_area
    row_count = int(candidate.get("raw_row_count") or 0)
    col_count = int(candidate.get("n_cols") or 0)

    score = 0.0
    score += x_overlap * 3.0
    score -= gap / 180.0
    score -= max(0.0, area_ratio - 0.35) * 2.2
    score += min(1.0, row_count / 18.0) * 0.35
    score += min(1.0, col_count / 10.0) * 0.25
    return score


def _build_caption_guided_text_candidates(
    page: Any,
    caption_blocks: List[Dict[str, Any]],
    *,
    min_area: float,
    min_cols: int,
) -> List[Dict[str, Any]]:
    if not caption_blocks:
        return []

    x_margin = max(0.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_X_MARGIN_PT", "8"), 8.0))
    max_above = max(24.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MAX_ABOVE_PT", "320"), 320.0))
    max_below = max(24.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MAX_BELOW_PT", "260"), 260.0))
    min_clip_height = max(20.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MIN_CLIP_HEIGHT_PT", "48"), 48.0))
    max_candidates_per_caption = max(1, _safe_int(os.getenv("TABLE_TEXT_FALLBACK_MAX_CANDIDATES_PER_CAPTION", "1"), 1))

    page_bounds = _page_bbox(page)
    clip_x0 = min(page_bounds["x1"], max(page_bounds["x0"], page_bounds["x0"] + x_margin))
    clip_x1 = max(page_bounds["x0"], min(page_bounds["x1"], page_bounds["x1"] - x_margin))
    if clip_x1 <= clip_x0:
        clip_x0 = page_bounds["x0"]
        clip_x1 = page_bounds["x1"]

    all_candidates: List[Dict[str, Any]] = []
    for idx, caption in enumerate(caption_blocks):
        caption_bbox = caption.get("bbox")
        caption_text = str(caption.get("text") or "")
        if not caption_bbox:
            continue

        prev_caption_bbox = caption_blocks[idx - 1].get("bbox") if idx > 0 else None
        next_caption_bbox = caption_blocks[idx + 1].get("bbox") if idx + 1 < len(caption_blocks) else None

        clip_bboxes: List[Dict[str, float]] = []

        above_bottom = float(caption_bbox["y0"]) - 2.0
        above_floor = float(page_bounds["y0"]) + 2.0
        if prev_caption_bbox:
            above_floor = max(above_floor, float(prev_caption_bbox["y1"]) + 2.0)
        above_top = max(above_floor, above_bottom - max_above)
        if above_bottom - above_top >= min_clip_height:
            clip_bboxes.append({"x0": clip_x0, "y0": above_top, "x1": clip_x1, "y1": above_bottom})

        below_top = float(caption_bbox["y1"]) + 2.0
        below_ceiling = float(page_bounds["y1"]) - 2.0
        if next_caption_bbox:
            below_ceiling = min(below_ceiling, float(next_caption_bbox["y0"]) - 2.0)
        below_bottom = min(below_ceiling, below_top + max_below)
        if below_bottom - below_top >= min_clip_height:
            clip_bboxes.append({"x0": clip_x0, "y0": below_top, "x1": clip_x1, "y1": below_bottom})

        caption_candidates: List[Dict[str, Any]] = []
        for clip_bbox in clip_bboxes:
            clip_rect = _bbox_to_rect_tuple(clip_bbox)
            if not clip_rect:
                continue
            try:
                finder = page.find_tables(strategy="text", clip=clip_rect)
            except Exception:
                continue
            text_tables = getattr(finder, "tables", None) or []
            if not text_tables:
                continue
            caption_candidates.extend(
                _build_candidates_from_table_objects(
                    text_tables,
                    min_area=min_area,
                    min_cols=min_cols,
                    detection_strategy="text_caption_fallback",
                    seed_caption=caption_text,
                    seed_caption_id=idx,
                    seed_caption_bbox=caption_bbox,
                    clip_bbox=clip_bbox,
                )
            )

        if not caption_candidates:
            continue
        caption_candidates = _merge_table_candidates(caption_candidates)
        caption_candidates.sort(
            key=lambda item: _score_text_fallback_candidate(
                item,
                caption_bbox=caption_bbox,
                page_bbox=page_bounds,
            ),
            reverse=True,
        )
        all_candidates.extend(caption_candidates[:max_candidates_per_caption])

    return _merge_table_candidates(all_candidates)


def _default_table_dir() -> Path:
    root = (Path.cwd() / ".ia_phase1_data" / "tables").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _table_root() -> Path:
    configured = os.getenv("TABLE_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _default_table_dir()


def _paper_dir(paper_id: int) -> Path:
    return _table_root() / str(int(paper_id))


def _manifest_path(paper_id: int) -> Path:
    return _paper_dir(paper_id) / "manifest.json"


def _table_enabled() -> bool:
    raw = os.getenv("TABLE_EXTRACTION_ENABLED", "true").strip().lower()
    return raw in {"1", "true", "yes"}


def _prepare_page_blocks(blocks: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    page_map: Dict[int, List[Dict[str, Any]]] = {}
    for block in blocks:
        page_no = _safe_int(block.get("page_no"), 0)
        if page_no <= 0:
            continue
        block_meta_raw = block.get("metadata")
        block_meta = block_meta_raw if isinstance(block_meta_raw, dict) else {}
        page_map.setdefault(page_no, []).append(
            {
                "page_no": page_no,
                "block_index": _safe_int(block.get("block_index"), 0),
                "bbox": _bbox_from_payload(block.get("bbox")),
                "section_canonical": str(block_meta.get("section_canonical") or "other"),
                "section_title": str(block_meta.get("section_title") or "Document Body"),
                "section_source": str(block_meta.get("section_source") or "fallback"),
                "section_confidence": _safe_float(block_meta.get("section_confidence"), 0.35),
            }
        )
    for page_no, page_blocks in page_map.items():
        page_blocks.sort(key=lambda item: item.get("block_index", 0))
        page_map[page_no] = page_blocks
    return page_map


def _infer_section_for_table(
    page_blocks: List[Dict[str, Any]],
    table_bbox: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    if not page_blocks:
        return {
            "section_canonical": "other",
            "section_title": "Document Body",
            "section_source": "fallback",
            "section_confidence": 0.25,
        }

    best_block: Optional[Dict[str, Any]] = None
    best_score = -1.0
    table_area = max(_rect_area(table_bbox), 1e-6)

    for block in page_blocks:
        block_bbox = block.get("bbox")
        overlap = _rect_overlap(table_bbox, block_bbox)
        score = 0.0
        if overlap > 0:
            block_area = max(_rect_area(block_bbox), 1e-6)
            score = (overlap / table_area) * 0.8 + (overlap / block_area) * 0.2
        elif table_bbox and block_bbox:
            distance = abs(_center_y(table_bbox) - _center_y(block_bbox))
            score = 0.04 / (1.0 + distance)
        else:
            score = 0.01
        if score > best_score:
            best_score = score
            best_block = block

    if not best_block:
        return {
            "section_canonical": "other",
            "section_title": "Document Body",
            "section_source": "fallback",
            "section_confidence": 0.25,
        }

    confidence = max(
        0.25,
        min(
            0.98,
            _safe_float(best_block.get("section_confidence"), 0.35) * (0.8 + max(0.0, best_score)),
        ),
    )
    return {
        "section_canonical": str(best_block.get("section_canonical") or "other"),
        "section_title": str(best_block.get("section_title") or "Document Body"),
        "section_source": str(best_block.get("section_source") or "fallback"),
        "section_confidence": round(confidence, 3),
    }


def _clean_cell(value: Any, *, keep_newlines: bool = False) -> str:
    text = str(value or "").replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if keep_newlines:
        lines = [" ".join(line.split()).strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines).strip()
    text = " ".join(text.split())
    return text.strip()


def _normalize_matrix(raw_rows: Any) -> List[List[str]]:
    if not isinstance(raw_rows, list):
        return []
    matrix: List[List[str]] = []
    max_cols = 0
    for row in raw_rows:
        if not isinstance(row, (list, tuple)):
            continue
        cleaned = [_clean_cell(cell, keep_newlines=True) for cell in row]
        if not any(cleaned):
            continue
        matrix.append(cleaned)
        max_cols = max(max_cols, len(cleaned))

    if not matrix or max_cols <= 0:
        return []

    for row in matrix:
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))
        elif len(row) > max_cols:
            del row[max_cols:]
    return matrix


def _pick_headers_and_rows(matrix: List[List[str]], table_obj: Any) -> Tuple[List[str], List[List[str]]]:
    header_names: List[str] = []
    header_obj = getattr(table_obj, "header", None)
    header_candidates = getattr(header_obj, "names", None) if header_obj is not None else None
    if isinstance(header_candidates, (list, tuple)):
        header_names = [_clean_cell(item) for item in header_candidates]
        if not any(header_names):
            header_names = []

    if not matrix:
        return header_names, []

    n_cols = len(matrix[0])
    if header_names and len(header_names) < n_cols:
        header_names.extend([""] * (n_cols - len(header_names)))
    if header_names and len(header_names) > n_cols:
        header_names = header_names[:n_cols]

    if header_names:
        return header_names, matrix

    first = matrix[0]
    if len(matrix) >= 2 and sum(1 for cell in first if cell) >= max(1, len(first) // 2):
        return first, matrix[1:]

    generated = [f"col_{idx + 1}" for idx in range(n_cols)]
    return generated, matrix


def _to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    if not headers:
        return ""

    def _escape(cell: str) -> str:
        return (cell or "").replace("|", "\\|").replace("\n", "<br/>")

    header_row = "| " + " | ".join(_escape(cell) for cell in headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = [
        "| " + " | ".join(_escape(cell) for cell in row[:len(headers)]) + " |"
        for row in rows
    ]
    return "\n".join([header_row, divider, *body_rows]).strip()


def _to_csv_text(headers: List[str], rows: List[List[str]]) -> str:
    if not headers:
        return ""
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    for row in rows:
        normalized = row[:len(headers)] + [""] * max(0, len(headers) - len(row))
        writer.writerow(normalized[:len(headers)])
    return buffer.getvalue().strip()


def _extract_block_text(block: Dict[str, Any]) -> str:
    lines = block.get("lines") or []
    parts: List[str] = []
    for line in lines:
        spans = line.get("spans") or []
        seg = "".join(str(span.get("text") or "") for span in spans)
        seg = " ".join(seg.split()).strip()
        if seg:
            parts.append(seg)
    return " ".join(parts).strip()


def _find_caption_with_pattern(
    page: Any,
    target_bbox: Optional[Dict[str, float]],
    pattern: re.Pattern[str],
    *,
    prefer_above: bool = True,
    max_distance_pt: float = 120.0,
    fallback_nearby: bool = False,
) -> Optional[str]:
    if not target_bbox:
        return None
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return None

    target_top = float(target_bbox["y0"])
    target_bottom = float(target_bbox["y1"])
    candidates: List[Tuple[float, str]] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = _bbox_from_tuple(block.get("bbox"))
        if not bbox:
            continue
        text = _extract_block_text(block)
        if not text:
            continue

        block_top = float(bbox["y0"])
        block_bottom = float(bbox["y1"])
        side: Optional[str] = None
        distance = 0.0
        if block_bottom <= target_top + 2.0:
            distance = target_top - block_bottom
            side = "above"
        elif block_top >= target_bottom - 2.0:
            distance = block_top - target_bottom
            side = "below"
        if side is None:
            continue
        if distance > max_distance_pt:
            continue

        if pattern.search(text):
            penalty = 0.0
            if prefer_above and side == "below":
                penalty = 20.0
            elif (not prefer_above) and side == "above":
                penalty = 20.0
            candidates.append((distance + penalty, text[:260]))
            continue

        if fallback_nearby and side == "above" and distance <= 24.0:
            candidates.append((distance + 60.0, text[:180]))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _find_table_caption(page: Any, table_bbox: Optional[Dict[str, float]]) -> Optional[str]:
    max_distance = _safe_float(os.getenv("TABLE_CAPTION_MAX_DISTANCE_PT", "120"), 120.0)
    return _find_caption_with_pattern(
        page,
        table_bbox,
        _TABLE_CAPTION_RE,
        prefer_above=True,
        max_distance_pt=max_distance,
        fallback_nearby=True,
    )


def _find_figure_caption(page: Any, table_bbox: Optional[Dict[str, float]]) -> Optional[str]:
    max_distance = _safe_float(os.getenv("TABLE_CAPTION_MAX_DISTANCE_PT", "120"), 120.0)
    return _find_caption_with_pattern(
        page,
        table_bbox,
        _FIGURE_CAPTION_RE,
        prefer_above=False,
        max_distance_pt=max_distance,
        fallback_nearby=False,
    )


def _flatten_matrix_cells(matrix: List[List[str]]) -> List[str]:
    cells: List[str] = []
    for row in matrix:
        for cell in row:
            text = _clean_cell(cell)
            if text:
                cells.append(text)
    return cells


def _row_signature(row: List[str]) -> str:
    return "|".join(_clean_cell(cell).lower() for cell in row).strip()


def _looks_like_false_positive_table(
    matrix: List[List[str]],
    *,
    n_cols: int,
    row_count: int,
    table_caption: Optional[str],
    figure_caption: Optional[str],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    has_table_caption = bool(table_caption and _TABLE_CAPTION_RE.search(table_caption))
    has_figure_caption = bool(figure_caption and _FIGURE_CAPTION_RE.search(figure_caption))
    if has_figure_caption and not has_table_caption:
        reasons.append("nearby_figure_caption")
        return True, reasons

    cells = _flatten_matrix_cells(matrix)
    if not cells:
        return True, ["empty_cells"]

    joined = " ".join(cells)
    tokens = re.findall(r"[A-Za-z0-9_]+", joined)
    numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    numeric_ratio = (numeric_tokens / len(tokens)) if tokens else 0.0

    avg_cell_chars = sum(len(cell) for cell in cells) / max(1, len(cells))
    long_cell_ratio = sum(1 for cell in cells if len(cell) >= 40) / max(1, len(cells))
    duplicate_row_ratio = 0.0
    if matrix:
        signatures = [_row_signature(row) for row in matrix]
        signatures = [sig for sig in signatures if sig]
        if signatures:
            unique_count = len(set(signatures))
            duplicate_row_ratio = 1.0 - (unique_count / len(signatures))

    if _BASE64_LIKE_RE.search(joined) and not has_table_caption:
        reasons.append("base64_like_noise")
    if row_count <= 2 and n_cols >= 8 and not has_table_caption:
        reasons.append("very_wide_shallow_without_table_caption")
    if avg_cell_chars > 30.0 and numeric_ratio < 0.05 and not has_table_caption:
        reasons.append("prose_like_cells_without_table_caption")
    if long_cell_ratio > 0.45 and numeric_ratio < 0.08 and not has_table_caption:
        reasons.append("too_many_long_cells_low_numeric")
    if duplicate_row_ratio > 0.45 and n_cols >= 6 and not has_table_caption:
        reasons.append("duplicate_rows_pattern")

    min_false_positive_signals = max(1, _safe_int(os.getenv("TABLE_FALSE_POSITIVE_MIN_SIGNALS", "2"), 2))
    if len(reasons) >= min_false_positive_signals:
        return True, reasons
    return False, []


def extract_and_store_paper_tables(
    pdf_path: Path,
    paper_id: int,
    blocks: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract structured tables and store a per-paper manifest.
    """
    if not _table_enabled():
        return {"paper_id": int(paper_id), "num_tables": 0, "manifest_path": None, "tables": []}

    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_blocks = _prepare_page_blocks(blocks)
    output_dir = _paper_dir(paper_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale in output_dir.glob("table_*.json"):
        try:
            stale.unlink()
        except Exception:
            logger.warning("Failed to remove stale table file %s", stale)

    table_records: List[Dict[str, Any]] = []
    min_rows = max(1, _safe_int(os.getenv("TABLE_MIN_ROWS", "2"), 2))
    min_cols = max(1, _safe_int(os.getenv("TABLE_MIN_COLS", "2"), 2))
    min_area = max(1.0, _safe_float(os.getenv("TABLE_MIN_AREA_PT", "1600"), 1600.0))
    text_fallback_enabled = os.getenv("TABLE_TEXT_FALLBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    dedup_iou_threshold = min(1.0, max(0.0, _safe_float(os.getenv("TABLE_DEDUP_IOU_THRESHOLD", "0.80"), 0.80)))

    doc = pymupdf.open(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_no = page_index + 1
            if not hasattr(page, "find_tables"):
                logger.warning("PyMuPDF build does not support page.find_tables(); skipping table extraction.")
                break

            page_bounds = _page_bbox(page)
            page_kept_bboxes: List[Optional[Dict[str, float]]] = []
            page_candidates: List[Dict[str, Any]] = []

            try:
                finder = page.find_tables(strategy="lines")
            except Exception as exc:
                logger.warning("Table detection failed for paper %s page %s: %s", paper_id, page_no, exc)
                continue

            page_tables = getattr(finder, "tables", None) or []
            page_candidates.extend(
                _build_candidates_from_table_objects(
                    page_tables,
                    min_area=min_area,
                    min_cols=min_cols,
                    detection_strategy="lines",
                )
            )

            if text_fallback_enabled:
                table_caption_blocks = _collect_caption_blocks(page, _TABLE_CAPTION_RE)
                if table_caption_blocks:
                    page_candidates.extend(
                        _build_caption_guided_text_candidates(
                            page,
                            table_caption_blocks,
                            min_area=min_area,
                            min_cols=min_cols,
                        )
                    )

            for candidate in _merge_table_candidates(page_candidates):
                bbox = candidate.get("bbox")
                matrix = candidate.get("matrix") or []
                table_obj = candidate.get("table_obj")
                detection_strategy = str(candidate.get("detection_strategy") or "lines")
                if not matrix:
                    continue

                if _bbox_matches_any_iou(bbox, page_kept_bboxes, dedup_iou_threshold):
                    continue

                headers, rows = _pick_headers_and_rows(matrix, table_obj)
                n_cols = len(headers) if headers else len(matrix[0])
                matrix_row_count = len(matrix)
                row_count = len(rows)
                if n_cols < min_cols:
                    continue
                if matrix_row_count < min_rows:
                    continue

                seed_caption_text = str(candidate.get("seed_caption") or "").strip()
                seed_caption_bbox = candidate.get("seed_caption_bbox")
                if detection_strategy == "text_caption_fallback":
                    if not _passes_text_fallback_constraints(bbox, page_bounds, seed_caption_bbox):
                        logger.info(
                            "Skipping text fallback candidate for paper %s page %s (%sx%s): failed local caption constraints",
                            paper_id,
                            page_no,
                            row_count,
                            n_cols,
                        )
                        continue

                table_id = len(table_records) + 1
                caption = _find_table_caption(page, bbox)
                if detection_strategy == "text_caption_fallback" and not caption and seed_caption_text:
                    caption = seed_caption_text[:260]
                if detection_strategy == "text_caption_fallback" and not (caption and _TABLE_CAPTION_RE.search(caption)):
                    logger.info(
                        "Skipping text fallback candidate for paper %s page %s (%sx%s): no nearby table caption",
                        paper_id,
                        page_no,
                        row_count,
                        n_cols,
                    )
                    continue

                figure_caption = _find_figure_caption(page, bbox)
                is_false_positive, fp_reasons = _looks_like_false_positive_table(
                    matrix,
                    n_cols=n_cols,
                    row_count=row_count,
                    table_caption=caption,
                    figure_caption=figure_caption,
                )
                if is_false_positive:
                    logger.info(
                        "Skipping table candidate for paper %s page %s (%sx%s, fragments=%s, strategy=%s): %s",
                        paper_id,
                        page_no,
                        row_count,
                        n_cols,
                        int(candidate.get("merged_fragments") or 1),
                        detection_strategy,
                        ", ".join(fp_reasons),
                    )
                    continue

                section = _infer_section_for_table(page_blocks.get(page_no, []), bbox)
                markdown = _to_markdown(headers, rows)
                csv_text = _to_csv_text(headers, rows)

                record = {
                    "id": table_id,
                    "paper_id": int(paper_id),
                    "page_no": page_no,
                    "bbox": bbox,
                    "caption": caption,
                    "n_rows": len(rows),
                    "n_cols": n_cols,
                    "headers": headers,
                    "rows": rows,
                    "markdown": markdown,
                    "csv_text": csv_text,
                    "section_canonical": section["section_canonical"],
                    "section_title": section["section_title"],
                    "section_source": section["section_source"],
                    "section_confidence": section["section_confidence"],
                    "figure_caption_nearby": figure_caption,
                    "merged_fragments": int(candidate.get("merged_fragments") or 1),
                    "detection_strategy": detection_strategy,
                    "json_file": f"table_{table_id:04d}.json",
                }
                table_path = output_dir / record["json_file"]
                with table_path.open("w", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                table_records.append(record)
                page_kept_bboxes.append(bbox)
    finally:
        doc.close()

    manifest = {
        "paper_id": int(paper_id),
        "source_pdf": str(pdf_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_tables": len(table_records),
        "tables": table_records,
    }
    manifest_path = _manifest_path(paper_id)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return {
        "paper_id": int(paper_id),
        "num_tables": len(table_records),
        "manifest_path": str(manifest_path),
        "tables": table_records,
    }


def _next_indices_by_page(text_blocks: Iterable[Dict[str, Any]]) -> Dict[int, int]:
    max_by_page: Dict[int, int] = {}
    for block in text_blocks:
        page_no = _safe_int(block.get("page_no"), 0)
        if page_no <= 0:
            continue
        idx = _safe_int(block.get("block_index"), 0)
        max_by_page[page_no] = max(max_by_page.get(page_no, -1), idx)
    next_by_page: Dict[int, int] = {}
    for page_no, max_idx in max_by_page.items():
        next_by_page[page_no] = max_idx + 1000
    return next_by_page


def table_records_to_chunks(
    tables: Iterable[Dict[str, Any]],
    text_blocks: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert structured tables into chunk records for pgvector insertion.
    """
    records = [item for item in tables if isinstance(item, dict)]
    if not records:
        return []

    max_rows = max(1, _safe_int(os.getenv("TABLE_MAX_ROWS_PER_CHUNK", "12"), 12))
    max_chars = max(500, _safe_int(os.getenv("TABLE_CHUNK_MAX_CHARS", "2200"), 2200))
    next_by_page = _next_indices_by_page(text_blocks)
    chunks: List[Dict[str, Any]] = []

    for table in records:
        page_no = _safe_int(table.get("page_no"), 1)
        headers_raw = table.get("headers")
        headers = [str(cell) for cell in headers_raw] if isinstance(headers_raw, list) else []
        rows_raw = table.get("rows")
        rows = rows_raw if isinstance(rows_raw, list) else []
        rows = [row for row in rows if isinstance(row, list)]
        if not headers:
            if rows:
                headers = [f"col_{idx + 1}" for idx in range(len(rows[0]))]
            else:
                continue

        table_id = _safe_int(table.get("id"), 0)
        caption = str(table.get("caption") or f"Table {table_id}").strip()
        section_canonical = str(table.get("section_canonical") or "other").strip() or "other"
        section_title = str(table.get("section_title") or "Document Body").strip() or "Document Body"
        section_source = str(table.get("section_source") or "fallback").strip() or "fallback"
        section_confidence = _safe_float(table.get("section_confidence"), 0.35)
        table_bbox = _bbox_from_payload(table.get("bbox"))

        if page_no not in next_by_page:
            next_by_page[page_no] = 1000

        if not rows:
            rows = [[]]

        row_ptr = 0
        while row_ptr < len(rows):
            if rows == [[]]:
                window = []
                consumed = 1
            else:
                window = rows[row_ptr:row_ptr + max_rows]
                consumed = max(1, len(window))

            row_start = row_ptr + 1 if rows != [[]] else 0
            row_end = row_ptr + consumed if rows != [[]] else 0
            markdown = _to_markdown(headers, window)
            text_parts = [
                caption,
                f"Section: {section_title}",
                markdown,
            ]
            chunk_text = "\n\n".join(part for part in text_parts if part).strip()

            while len(chunk_text) > max_chars and consumed > 1:
                consumed -= 1
                window = window[:consumed]
                row_end = row_ptr + consumed
                markdown = _to_markdown(headers, window)
                chunk_text = "\n\n".join(part for part in [caption, f"Section: {section_title}", markdown] if part).strip()

            block_index = next_by_page[page_no]
            next_by_page[page_no] = block_index + 1

            source_block = {
                "page_no": page_no,
                "block_index": block_index,
                "text": chunk_text,
                "bbox": table_bbox,
                "metadata": {
                    "section_title": section_title,
                    "section_canonical": section_canonical,
                    "section_source": section_source,
                    "section_confidence": round(section_confidence, 3),
                },
            }

            metadata = {
                "chunk_type": "table_rows",
                "content_type": "table",
                "table_id": table_id,
                "table_caption": caption,
                "table_page_no": page_no,
                "table_row_start": row_start,
                "table_row_end": row_end,
                "table_total_rows": _safe_int(table.get("n_rows"), len(rows)),
                "table_total_cols": _safe_int(table.get("n_cols"), len(headers)),
                "table_columns": headers,
                "table_chunk_index": max(0, row_start - 1) // max_rows if row_start > 0 else 0,
                "section_primary": section_canonical,
                "section_all": [section_canonical],
                "section_titles": [section_title],
                "section_source": section_source,
                "section_confidence": round(section_confidence, 3),
                "spans_multiple_sections": False,
                "blocks": [source_block],
            }

            chunks.append(
                {
                    "text": chunk_text,
                    "page_no": page_no,
                    "block_index": block_index,
                    "bbox": table_bbox,
                    "metadata": metadata,
                }
            )

            row_ptr += consumed
            if rows == [[]]:
                break

    return chunks


def load_paper_table_manifest(paper_id: int) -> Dict[str, Any]:
    path = _manifest_path(paper_id)
    if not path.exists():
        return {"paper_id": int(paper_id), "num_tables": 0, "tables": []}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read table manifest for paper %s: %s", paper_id, exc)
        return {"paper_id": int(paper_id), "num_tables": 0, "tables": []}
    if not isinstance(payload, dict):
        return {"paper_id": int(paper_id), "num_tables": 0, "tables": []}
    tables = payload.get("tables")
    if not isinstance(tables, list):
        tables = []
    payload["tables"] = tables
    payload["num_tables"] = _safe_int(payload.get("num_tables"), len(tables))
    payload["paper_id"] = _safe_int(payload.get("paper_id"), int(paper_id))
    return payload