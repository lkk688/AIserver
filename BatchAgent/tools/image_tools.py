"""
image_tools.py

Safe image loading, resizing, and base64-encoding utilities for the agent.

Public API
----------
process_and_encode_image(source, max_dimension) -> data URI str
    source may be a local file path OR an http(s) URL.

execute_view_image(path, tool_call_id)        -> OpenAI-compatible tool response dict
make_view_image_router_result(path)           -> sentinel string for ToolRouter.dispatch()

The sentinel mechanism lets ToolRouter (which always returns plain str) carry
multimodal payload through to the agent without breaking the existing string
pipeline.  The agent detects the sentinel prefix, extracts the image blocks,
and attaches them to the next user message content list.

Supported formats
-----------------
Local files: JPEG, JPG, PNG, GIF, BMP, WEBP, TIFF, HEIC/HEIF (requires pillow-heif)
Remote URLs: any of the above fetched over http/https
All modes (RGBA, P, L …) are normalised to RGB before JPEG encoding.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# ── Sentinel prefix ────────────────────────────────────────────────────────────
# Chosen to be invisible in normal text and unlikely to appear in real output.
_MULTIMODAL_SENTINEL = "\x00MM\x00"

# ── Supported image extensions ─────────────────────────────────────────────────
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".webp", ".tiff", ".tif", ".heic", ".heif", ".avif",
}


def is_image_path(path: str) -> bool:
    """Return True if *path* (file path or URL) has a recognised image extension."""
    ext = os.path.splitext(urlparse(path).path)[1].lower()
    return ext in IMAGE_EXTENSIONS


# ── PIL is required ────────────────────────────────────────────────────────────
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ── HEIC/HEIF support via pillow-heif (optional) ──────────────────────────────
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False


# =============================================================================
# 1. Image loading helpers
# =============================================================================

_IMAGE_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _fetch_url_bytes(url: str, timeout: int = 20) -> bytes:
    """
    Fetch raw bytes from an http/https URL.

    Uses urllib (stdlib) as the primary transport — it passes Wikimedia's
    bot-detection while httpx (with identical headers) gets blocked.
    Falls back to httpx if urllib raises an unexpected error.

    A browser-like User-Agent and a Referer derived from the image origin
    are sent on every request.
    """
    import urllib.request
    import urllib.error
    from urllib.parse import urlparse as _urlparse

    parsed = _urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    headers = {**_IMAGE_FETCH_HEADERS, "Referer": origin + "/"}

    # Primary: urllib (passes Wikimedia CDN checks)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 401, 429, 503}:
            pass  # try httpx fallback below
        else:
            raise

    # Fallback: httpx (may work for CDNs that urllib doesn't handle)
    try:
        import httpx
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.content
    except ImportError:
        raise RuntimeError(f"Cannot fetch image from {url}: urllib got 403 and httpx is not installed")


def _open_image_from_source(source: str) -> "Image.Image":
    """
    Open a PIL Image from either a local path or an http(s) URL.

    Supports HEIC/HEIF if pillow-heif is installed.
    Raises ValueError on any error.
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required for view_image. Install: pip install Pillow")

    ext = os.path.splitext(urlparse(source).path)[1].lower()
    is_url = source.startswith("http://") or source.startswith("https://")

    # ── HEIC / HEIF ───────────────────────────────────────────────────────────
    if ext in {".heic", ".heif"}:
        if not _HEIF_AVAILABLE:
            raise ImportError(
                "pillow-heif is required for HEIC/HEIF images. "
                "Install: pip install pillow-heif"
            )
        if is_url:
            data = _fetch_url_bytes(source)
            return Image.open(io.BytesIO(data))
        return Image.open(source)

    # ── Remote URL ────────────────────────────────────────────────────────────
    if is_url:
        data = _fetch_url_bytes(source)
        return Image.open(io.BytesIO(data))

    # ── Local file ────────────────────────────────────────────────────────────
    return Image.open(source)


# =============================================================================
# 2. Core encode function
# =============================================================================

_ADAPTIVE_DIMENSIONS = [1024, 768, 512, 384, 256, 192, 128]


def process_and_encode_image(source: str, max_dimension: int = 1024) -> str:
    """
    Load, normalise, resize, and base64-encode an image.

    ``source`` may be:
    - A local file path (absolute or relative)
    - An http/https URL

    Supported types: JPEG, PNG, GIF, BMP, WEBP, TIFF, HEIC/HEIF, AVIF

    Steps
    -----
    1. Convert any mode (RGBA, P, L …) to RGB.
    2. Resize the longest edge to ``max_dimension`` (LANCZOS), keeping aspect ratio.
    3. Re-encode to JPEG at quality=85 in memory.
    4. Return a ``data:image/jpeg;base64,<b64>`` data URI.

    Raises
    ------
    ValueError  — if the image cannot be opened / decoded.
    ImportError — if Pillow (or pillow-heif for HEIC) is missing.
    """
    try:
        with _open_image_from_source(source) as img:
            # 1. Normalise to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 2. Resize if too large
            if max(img.width, img.height) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            # 3. JPEG encode into memory
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)

            # 4. Base64 → data URI
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"

    except (ImportError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Failed to process image '{source}': {exc}") from exc


def process_and_encode_image_adaptive(
    source: str,
    max_tokens: int = 8192,
) -> tuple:
    """
    Encode an image, trying progressively smaller resolutions until the
    estimated token cost (data-URI length ÷ 4) fits within ``max_tokens``.

    Returns
    -------
    (data_uri, actual_dimension, estimated_tokens)

    The smallest dimension tried is 128 px.  If even 128 px exceeds the budget
    the smallest result is still returned — callers handle residual overflow.
    """
    best_uri, best_dim, best_tokens = None, 128, max_tokens + 1
    for dim in _ADAPTIVE_DIMENSIONS:
        uri = process_and_encode_image(source, max_dimension=dim)
        tok = max(256, len(uri) // 4)
        best_uri, best_dim, best_tokens = uri, dim, tok
        if tok <= max_tokens:
            break
    return best_uri, best_dim, best_tokens


# =============================================================================
# 3. Native tool-call response builder (for OpenAI tool-call result messages)
# =============================================================================

def execute_view_image(
    path: str,
    tool_call_id: str = "",
    max_dimension: int = 1024,
) -> Dict[str, Any]:
    """
    Build the OpenAI-style tool response dict for a ``view_image`` call.

    Returns a dict with ``role="tool"``, ``tool_call_id``, and a multimodal
    ``content`` list containing a text description and the image_url block.

    This is the correct response format when the model used *native* tool
    calling (not text/XML hybrid).  For the text-based hybrid path, use
    ``make_view_image_router_result()`` instead.
    """
    is_url = path.startswith("http://") or path.startswith("https://")
    if not is_url and not os.path.exists(path):
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": f"Error: Image file not found: '{path}'",
        }
    try:
        data_uri = process_and_encode_image(path, max_dimension=max_dimension)
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Successfully loaded image from '{path}'. "
                        "Please analyze it carefully."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                },
            ],
        }
    except Exception as exc:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": f"Error processing image '{path}': {exc}",
        }


# =============================================================================
# 4. ToolRouter sentinel builder (for text/hybrid dispatch path)
# =============================================================================

def make_view_image_router_result(
    path: str,
    max_tokens: int = 8192,
) -> str:
    """
    Build the string result returned by ``ToolRouter.dispatch("view_image", ...)``.

    ``path`` may be a local file path OR an http/https URL.

    Resolution is chosen adaptively so that the image's token cost fits within
    ``max_tokens``.  This prevents a single image from consuming the entire
    context window when the model has limited headroom.

    If the image loads successfully, returns a sentinel-prefixed JSON string:
        ``_MULTIMODAL_SENTINEL + json.dumps({"text": "...", "image_blocks": [...]})``

    The agent detects this prefix, extracts the image blocks, and appends them
    to the next ``user`` message's content list so the vision model can see them.

    On failure, returns a plain error string (no sentinel).
    """
    is_url = path.startswith("http://") or path.startswith("https://")
    if not is_url and not os.path.exists(path):
        return f"Error: Image file not found: '{path}'"
    try:
        data_uri, actual_dim, tok_cost = process_and_encode_image_adaptive(
            path, max_tokens=max_tokens
        )
        payload = {
            "text": (
                f"Successfully loaded image from '{path}' "
                f"(resized to {actual_dim}px, ~{tok_cost} tokens). "
                "The image has been attached below for your analysis."
            ),
            "image_blocks": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            ],
        }
        return _MULTIMODAL_SENTINEL + json.dumps(payload)
    except Exception as exc:
        return f"Error processing image '{path}': {exc}"


# =============================================================================
# 5. Helper: decode a sentinel string produced by make_view_image_router_result
# =============================================================================

def is_multimodal_sentinel(text: str) -> bool:
    """Return True if *text* is a sentinel-encoded multimodal result."""
    return isinstance(text, str) and text.startswith(_MULTIMODAL_SENTINEL)


def decode_multimodal_sentinel(text: str) -> Dict[str, Any]:
    """
    Decode a sentinel string.

    Returns
    -------
    dict with keys:
        ``text``         — plain text description shown inline in the feedback
        ``image_blocks`` — list of ``{"type": "image_url", ...}`` dicts
    """
    raw = text[len(_MULTIMODAL_SENTINEL):]
    return json.loads(raw)
