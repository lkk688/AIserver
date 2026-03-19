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

_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.)\s*\d+\b", re.IGNORECASE)


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


def _bbox_from_rect(rect: Any) -> Optional[Dict[str, float]]:
    if rect is None:
        return None
    try:
        return {
            "x0": float(rect.x0),
            "y0": float(rect.y0),
            "x1": float(rect.x1),
            "y1": float(rect.y1),
        }
    except Exception:
        return None


def _bbox_from_payload(payload: Any) -> Optional[Dict[str, float]]:
    if not isinstance(payload, dict):
        return None
    x0 = _safe_float(payload.get("x0"), float("nan"))
    y0 = _safe_float(payload.get("y0"), float("nan"))
    x1 = _safe_float(payload.get("x1"), float("nan"))
    y1 = _safe_float(payload.get("y1"), float("nan"))
    if any(v != v for v in (x0, y0, x1, y1)):  # NaN check
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _bbox_from_tuple(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return None
    x0 = _safe_float(value[0], float("nan"))
    y0 = _safe_float(value[1], float("nan"))
    x1 = _safe_float(value[2], float("nan"))
    y1 = _safe_float(value[3], float("nan"))
    if any(v != v for v in (x0, y0, x1, y1)):
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


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


def _center_y(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return (bbox["y0"] + bbox["y1"]) * 0.5


def _bbox_width(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox["x1"]) - float(bbox["x0"]))


def _bbox_height(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox["y1"]) - float(bbox["y0"]))


def _bbox_center_x(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return (float(bbox["x0"]) + float(bbox["x1"])) * 0.5


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


def _bbox_intersection(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not a or not b:
        return None
    ix0 = max(float(a["x0"]), float(b["x0"]))
    iy0 = max(float(a["y0"]), float(b["y0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    iy1 = min(float(a["y1"]), float(b["y1"]))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return {"x0": ix0, "y0": iy0, "x1": ix1, "y1": iy1}


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


def _clip_bbox_to_bounds(
    bbox: Optional[Dict[str, float]],
    bounds: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if not bbox or not bounds:
        return None
    return _bbox_intersection(bbox, bounds)


def _page_bounds(page: Any) -> Dict[str, float]:
    rect = page.rect
    return {
        "x0": float(rect.x0),
        "y0": float(rect.y0),
        "x1": float(rect.x1),
        "y1": float(rect.y1),
    }


def _line_text(line: Dict[str, Any]) -> str:
    spans = line.get("spans") or []
    text = "".join(str(span.get("text") or "") for span in spans)
    return " ".join(text.split()).strip()


def _extract_figure_captions_from_page(page: Any) -> List[Dict[str, Any]]:
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return []
    captions: List[Dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_bbox = _bbox_from_tuple(block.get("bbox"))
        for line in block.get("lines") or []:
            line_bbox = _bbox_from_tuple(line.get("bbox"))
            text = _line_text(line)
            if not text:
                continue
            if not _FIGURE_CAPTION_RE.search(text):
                continue
            anchor_bbox = line_bbox or block_bbox
            if not anchor_bbox:
                continue
            captions.append(
                {
                    "text": text[:320],
                    "bbox": anchor_bbox,
                    "block_bbox": block_bbox or anchor_bbox,
                }
            )
    captions.sort(key=lambda item: (float(item["bbox"]["y0"]), float(item["bbox"]["x0"])))
    return captions


def _collect_vector_drawing_bboxes(page: Any) -> List[Dict[str, float]]:
    min_side = max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_DRAWING_SIDE_PT", "0.8"), 0.8))
    drawings = page.get_drawings() or []
    results: List[Dict[str, float]] = []
    for drawing in drawings:
        bbox = _bbox_from_rect(drawing.get("rect"))
        if not bbox:
            continue
        width = _bbox_width(bbox)
        height = _bbox_height(bbox)
        if width < min_side and height < min_side:
            continue
        results.append(bbox)
    return results


def _bbox_x_overlap_ratio(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(float(a["x0"]), float(b["x0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    overlap = max(0.0, ix1 - ix0)
    base = max(1e-6, min(_bbox_width(a), _bbox_width(b)))
    return overlap / base


def _build_vector_region_from_caption(
    *,
    caption: Dict[str, Any],
    caption_index: int,
    captions: List[Dict[str, Any]],
    drawing_bboxes: List[Dict[str, float]],
    page_bounds: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    if not drawing_bboxes:
        return None

    caption_bbox = caption.get("bbox")
    if not isinstance(caption_bbox, dict):
        return None

    max_above = max(24.0, _safe_float(os.getenv("FIGURE_VECTOR_MAX_ABOVE_PT", "380"), 380.0))
    max_below = max(24.0, _safe_float(os.getenv("FIGURE_VECTOR_MAX_BELOW_PT", "240"), 240.0))
    min_window_height = max(16.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_WINDOW_HEIGHT_PT", "36"), 36.0))
    min_drawings = max(1, _safe_int(os.getenv("FIGURE_VECTOR_MIN_DRAWING_COUNT", "6"), 6))
    min_region_area = max(1.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_REGION_AREA_PT", "5000"), 5000.0))
    min_region_side = max(4.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_REGION_SIDE_PT", "40"), 40.0))
    margin = max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_MARGIN_PT", "6"), 6.0))
    column_pad = max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_COLUMN_X_PAD_PT", "28"), 28.0))
    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_X_OVERLAP_RATIO", "0.05"), 0.05)))

    page_width = max(1e-6, _bbox_width(page_bounds))
    caption_width = _bbox_width(caption_bbox)
    caption_is_wide = (caption_width / page_width) >= 0.72
    x0_gate = float(caption_bbox["x0"]) - column_pad
    x1_gate = float(caption_bbox["x1"]) + column_pad

    prev_caption = captions[caption_index - 1]["bbox"] if caption_index > 0 else None
    next_caption = captions[caption_index + 1]["bbox"] if caption_index + 1 < len(captions) else None

    windows: List[Tuple[str, Dict[str, float]]] = []
    above_bottom = float(caption_bbox["y0"]) - 2.0
    above_top = max(float(page_bounds["y0"]) + 1.0, above_bottom - max_above)
    if isinstance(prev_caption, dict):
        above_top = max(above_top, float(prev_caption["y1"]) + 2.0)
    if above_bottom - above_top >= min_window_height:
        windows.append(("above", {"x0": page_bounds["x0"], "y0": above_top, "x1": page_bounds["x1"], "y1": above_bottom}))

    below_top = float(caption_bbox["y1"]) + 2.0
    below_bottom = min(float(page_bounds["y1"]) - 1.0, below_top + max_below)
    if isinstance(next_caption, dict):
        below_bottom = min(below_bottom, float(next_caption["y0"]) - 2.0)
    if below_bottom - below_top >= min_window_height:
        windows.append(("below", {"x0": page_bounds["x0"], "y0": below_top, "x1": page_bounds["x1"], "y1": below_bottom}))

    best: Optional[Dict[str, Any]] = None
    for orientation, window_bbox in windows:
        selected: List[Dict[str, float]] = []
        for drawing_bbox in drawing_bboxes:
            clipped = _bbox_intersection(drawing_bbox, window_bbox)
            if not clipped:
                continue
            x_ok = caption_is_wide
            if not x_ok:
                if _bbox_x_overlap_ratio(clipped, caption_bbox) >= min_x_overlap:
                    x_ok = True
                else:
                    cx = _bbox_center_x(clipped)
                    x_ok = (cx >= x0_gate and cx <= x1_gate)
            if not x_ok:
                continue
            selected.append(clipped)
        if len(selected) < min_drawings:
            continue

        region_bbox: Optional[Dict[str, float]] = None
        for item in selected:
            region_bbox = _bbox_union(region_bbox, item)
        if not region_bbox:
            continue

        region_bbox = _clip_bbox_to_bounds(
            {
                "x0": float(region_bbox["x0"]) - margin,
                "y0": float(region_bbox["y0"]) - margin,
                "x1": float(region_bbox["x1"]) + margin,
                "y1": float(region_bbox["y1"]) + margin,
            },
            page_bounds,
        )
        if not region_bbox:
            continue
        if _rect_area(region_bbox) < min_region_area:
            continue
        if _bbox_width(region_bbox) < min_region_side or _bbox_height(region_bbox) < min_region_side:
            continue

        if orientation == "above":
            distance_penalty = max(0.0, float(caption_bbox["y0"]) - float(region_bbox["y1"]))
            orientation_bonus = 0.3
        else:
            distance_penalty = max(0.0, float(region_bbox["y0"]) - float(caption_bbox["y1"]))
            orientation_bonus = 0.0
        score = len(selected) * 0.25 + (_rect_area(region_bbox) / 30000.0) - (distance_penalty / 140.0) + orientation_bonus
        if not best or score > float(best.get("score") or -10**9):
            best = {
                "bbox": region_bbox,
                "score": score,
                "orientation": orientation,
                "drawing_count": len(selected),
            }

    return best


def _figure_root() -> Path:
    configured = os.getenv("FIGURE_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    root = (Path.cwd() / ".ia_phase1_data" / "figures").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _paper_dir(paper_id: int) -> Path:
    return _figure_root() / str(int(paper_id))


def _manifest_path(paper_id: int) -> Path:
    return _paper_dir(paper_id) / "manifest.json"


def _is_meaningful_image(width: int, height: int) -> bool:
    min_area = _safe_int(os.getenv("FIGURE_MIN_PIXEL_AREA", "4096"), 4096)
    min_side = _safe_int(os.getenv("FIGURE_MIN_SIDE_PX", "24"), 24)
    if width < min_side or height < min_side:
        return False
    return (width * height) >= max(1, min_area)


def _is_meaningful_render_rect(bbox: Optional[Dict[str, float]]) -> bool:
    if not bbox:
        return True
    width = max(0.0, float(bbox["x1"]) - float(bbox["x0"]))
    height = max(0.0, float(bbox["y1"]) - float(bbox["y0"]))
    min_side = _safe_float(os.getenv("FIGURE_MIN_RENDER_SIDE_PT", "18"), 18.0)
    min_area = _safe_float(os.getenv("FIGURE_MIN_RENDER_AREA", "800"), 800.0)
    if width < min_side or height < min_side:
        return False
    return (width * height) >= max(1.0, min_area)


def _prepare_page_blocks(blocks: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    page_map: Dict[int, List[Dict[str, Any]]] = {}
    for block in blocks:
        page_no = _safe_int(block.get("page_no"), 0)
        if page_no <= 0:
            continue
        metadata = block.get("metadata")
        block_meta = metadata if isinstance(metadata, dict) else {}
        section_canonical = str(block_meta.get("section_canonical") or "").strip() or "other"
        section_title = str(block_meta.get("section_title") or "").strip() or "Document Body"
        section_source = str(block_meta.get("section_source") or "").strip() or "fallback"
        section_confidence = _safe_float(block_meta.get("section_confidence"), 0.35)
        page_map.setdefault(page_no, []).append(
            {
                "page_no": page_no,
                "block_index": _safe_int(block.get("block_index"), 0),
                "bbox": _bbox_from_payload(block.get("bbox")),
                "section_canonical": section_canonical,
                "section_title": section_title,
                "section_source": section_source,
                "section_confidence": section_confidence,
            }
        )
    for page_no, page_blocks in page_map.items():
        page_blocks.sort(key=lambda item: item.get("block_index", 0))
        page_map[page_no] = page_blocks
    return page_map


def _infer_section_for_image(
    page_blocks: List[Dict[str, Any]],
    image_bbox: Optional[Dict[str, float]],
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
    image_area = max(_rect_area(image_bbox), 1e-6)

    for block in page_blocks:
        block_bbox = block.get("bbox")
        overlap = _rect_overlap(image_bbox, block_bbox)
        score = 0.0
        if overlap > 0:
            block_area = max(_rect_area(block_bbox), 1e-6)
            overlap_ratio_image = overlap / image_area
            overlap_ratio_block = overlap / block_area
            score = (overlap_ratio_image * 0.8) + (overlap_ratio_block * 0.2)
        elif image_bbox and block_bbox:
            # If no overlap, prefer nearest block vertically on the same page.
            distance = abs(_center_y(image_bbox) - _center_y(block_bbox))
            score = 0.05 / (1.0 + distance)
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

    inferred_confidence = max(
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
        "section_confidence": round(inferred_confidence, 3),
    }


def extract_and_store_paper_figures(
    pdf_path: Path,
    paper_id: int,
    blocks: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract embedded PDF figures for a paper and store a manifest mapped to sections.

    Returns a summary with extracted image count and manifest path.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_blocks = _prepare_page_blocks(blocks)
    output_dir = _paper_dir(paper_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale image files from previous extractions.
    for stale in output_dir.glob("page_*_img_*.*"):
        try:
            stale.unlink()
        except Exception:
            logger.warning("Failed to remove stale figure file %s", stale)
    for stale in output_dir.glob("page_*_vec_*.*"):
        try:
            stale.unlink()
        except Exception:
            logger.warning("Failed to remove stale figure file %s", stale)

    records: List[Dict[str, Any]] = []
    vector_enabled = os.getenv("FIGURE_VECTOR_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    vector_render_scale = max(0.5, _safe_float(os.getenv("FIGURE_VECTOR_RENDER_SCALE", "2.0"), 2.0))
    vector_dedup_iou = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_DEDUP_IOU", "0.82"), 0.82)))
    vector_skip_embedded_iou = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_EMBEDDED_IOU_SKIP", "0.88"), 0.88)))

    def _bbox_matches_any_iou(
        bbox: Optional[Dict[str, float]],
        candidates: Iterable[Optional[Dict[str, float]]],
        threshold: float,
    ) -> bool:
        if not bbox:
            return False
        for candidate in candidates:
            if _rect_iou(bbox, candidate) >= threshold:
                return True
        return False

    doc = pymupdf.open(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_no = page_index + 1
            page_record_bboxes: List[Optional[Dict[str, float]]] = []
            page_embedded_bboxes: List[Optional[Dict[str, float]]] = []
            images = page.get_images(full=True)
            for img_idx, image_info in enumerate(images, start=1):
                xref = _safe_int(image_info[0], -1)
                if xref <= 0:
                    continue
                try:
                    payload = doc.extract_image(xref)
                except Exception:
                    continue
                image_bytes = payload.get("image")
                if not image_bytes:
                    continue
                width = _safe_int(payload.get("width"), 0)
                height = _safe_int(payload.get("height"), 0)
                if width > 0 and height > 0 and not _is_meaningful_image(width, height):
                    continue

                ext = str(payload.get("ext") or "png").lower()
                file_name = f"page_{page_no:03d}_img_{img_idx:03d}.{ext}"
                image_path = output_dir / file_name
                with image_path.open("wb") as handle:
                    handle.write(image_bytes)

                rects = page.get_image_rects(xref) or []
                image_bbox = None
                if rects:
                    # Pick largest rendered rectangle for section assignment.
                    best_rect = max(rects, key=lambda rect: max(0.0, (rect.x1 - rect.x0)) * max(0.0, (rect.y1 - rect.y0)))
                    image_bbox = _bbox_from_rect(best_rect)
                if not _is_meaningful_render_rect(image_bbox):
                    continue

                if _bbox_matches_any_iou(image_bbox, page_record_bboxes, 0.98):
                    continue
                section = _infer_section_for_image(page_blocks.get(page_no, []), image_bbox)
                record = {
                    "id": len(records) + 1,
                    "paper_id": int(paper_id),
                    "page_no": page_no,
                    "file_name": file_name,
                    "image_path": str(image_path),
                    "url": f"/api/papers/{paper_id}/figures/{file_name}",
                    "width": width or None,
                    "height": height or None,
                    "bbox": image_bbox,
                    "section_canonical": section["section_canonical"],
                    "section_title": section["section_title"],
                    "section_source": section["section_source"],
                    "section_confidence": section["section_confidence"],
                    "figure_type": "embedded",
                    "figure_caption": None,
                }
                records.append(record)
                page_record_bboxes.append(image_bbox)
                page_embedded_bboxes.append(image_bbox)

            if not vector_enabled:
                continue

            captions = _extract_figure_captions_from_page(page)
            drawing_bboxes = _collect_vector_drawing_bboxes(page)
            page_bounds = _page_bounds(page)
            vec_idx = 0
            for caption_idx, caption in enumerate(captions):
                candidate = _build_vector_region_from_caption(
                    caption=caption,
                    caption_index=caption_idx,
                    captions=captions,
                    drawing_bboxes=drawing_bboxes,
                    page_bounds=page_bounds,
                )
                if not candidate:
                    continue
                region_bbox = candidate.get("bbox")
                if not _is_meaningful_render_rect(region_bbox):
                    continue
                if _bbox_matches_any_iou(region_bbox, page_embedded_bboxes, vector_skip_embedded_iou):
                    continue
                if _bbox_matches_any_iou(region_bbox, page_record_bboxes, vector_dedup_iou):
                    continue

                clip_rect = pymupdf.Rect(
                    float(region_bbox["x0"]),
                    float(region_bbox["y0"]),
                    float(region_bbox["x1"]),
                    float(region_bbox["y1"]),
                )
                try:
                    pix = page.get_pixmap(
                        matrix=pymupdf.Matrix(vector_render_scale, vector_render_scale),
                        clip=clip_rect,
                        alpha=False,
                    )
                except Exception:
                    continue

                vec_idx += 1
                file_name = f"page_{page_no:03d}_vec_{vec_idx:03d}.png"
                image_path = output_dir / file_name
                try:
                    pix.save(str(image_path))
                except Exception:
                    continue

                section = _infer_section_for_image(page_blocks.get(page_no, []), region_bbox)
                caption_text = str(caption.get("text") or "").strip() or None
                record = {
                    "id": len(records) + 1,
                    "paper_id": int(paper_id),
                    "page_no": page_no,
                    "file_name": file_name,
                    "image_path": str(image_path),
                    "url": f"/api/papers/{paper_id}/figures/{file_name}",
                    "width": int(getattr(pix, "width", 0)) or None,
                    "height": int(getattr(pix, "height", 0)) or None,
                    "bbox": region_bbox,
                    "section_canonical": section["section_canonical"],
                    "section_title": section["section_title"],
                    "section_source": section["section_source"],
                    "section_confidence": section["section_confidence"],
                    "figure_type": "vector",
                    "figure_caption": caption_text,
                    "vector_orientation": candidate.get("orientation"),
                    "vector_drawing_count": int(candidate.get("drawing_count") or 0),
                }
                records.append(record)
                page_record_bboxes.append(region_bbox)
    finally:
        doc.close()

    manifest = {
        "paper_id": int(paper_id),
        "source_pdf": str(pdf_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_images": len(records),
        "images": records,
    }
    manifest_path = _manifest_path(paper_id)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return {
        "paper_id": int(paper_id),
        "num_images": len(records),
        "manifest_path": str(manifest_path),
    }


def load_paper_figure_manifest(paper_id: int) -> Dict[str, Any]:
    path = _manifest_path(paper_id)
    if not path.exists():
        return {"paper_id": int(paper_id), "num_images": 0, "images": []}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read figure manifest for paper %s: %s", paper_id, exc)
        return {"paper_id": int(paper_id), "num_images": 0, "images": []}
    if not isinstance(payload, dict):
        return {"paper_id": int(paper_id), "num_images": 0, "images": []}
    images = payload.get("images")
    if not isinstance(images, list):
        images = []
    payload["images"] = images
    payload["num_images"] = _safe_int(payload.get("num_images"), len(images))
    payload["paper_id"] = _safe_int(payload.get("paper_id"), int(paper_id))
    return payload


def resolve_figure_file(paper_id: int, file_name: str) -> Path:
    safe_name = Path(file_name).name
    if safe_name != file_name:
        raise ValueError("Invalid figure file name.")
    path = _paper_dir(paper_id) / safe_name
    return path