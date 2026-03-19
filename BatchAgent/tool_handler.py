import re
import json
import html
import os
import tempfile
import urllib.request
import traceback
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote
from rich.console import Console
#pip install beautifulsoup4 markdownify httpx pypdf playwright
import httpx
from bs4 import BeautifulSoup
import markdownify

# Optional: pypdf for PDF extraction
try:
    from pypdf import PdfReader
    import io
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

# Optional: playwright for JS-heavy page fallback
try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

try:
    import yt_dlp
    _YTDLP_AVAILABLE = True
except ImportError:
    _YTDLP_AVAILABLE = False

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import necessary components from your codebase
# Make sure these are accessible or adjust imports as needed
from BatchAgent.mini_batch_agent_libs import (
    is_git_repo, apply_patch_guarded, apply_write_files, extract_all_diffs,
    extract_write_file_actions_v2, apply_fuzzy_patch, extract_files_from_diff,
    resolve_path, search_code, find_file, list_directory, read_file_chunk,
    run_bash_command, run_shell
)
from BatchAgent.mini_batch_agent_base import (
    AgentAction, ActionToolCall, ActionWriteFile, ActionApplyDiff, ActionReplaceText
)

console = Console()

# ---- Content cleaning constants ----
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside",
               "noscript", "iframe", "svg", "form", "button"]
_NOISE_PATTERNS = [
    re.compile(r'(cookie|gdpr|privacy|consent|subscribe|newsletter)', re.I),
    re.compile(r'^\s{0,4}[\|\-\–\•→✓✗★☆©®™]{1,3}\s*$'),  # icon-only lines
]
_MAX_CONTENT_CHARS = 15000
_PLAYWRIGHT_THRESHOLD = 300  # chars — below this, try playwright fallback


def _clean_text_content(text: str) -> str:
    """Shared post-processing: collapse blank lines, filter noise lines, truncate."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Drop very short lines that are likely nav/menu artifacts (< 3 words AND < 25 chars)
        if len(stripped) < 25 and len(stripped.split()) < 3 and stripped:
            continue
        # Drop lines matching noise patterns (cookie banners, GDPR boilerplate)
        if any(p.search(stripped) for p in _NOISE_PATTERNS):
            continue
        cleaned.append(line)

    text = '\n'.join(cleaned)
    # Collapse 3+ consecutive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    # Truncate
    if len(text) > _MAX_CONTENT_CHARS:
        text = text[:_MAX_CONTENT_CHARS] + "\n\n... [Content Truncated] ..."
    return text


def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Extract text from a PDF binary using pypdf."""
    if not _PYPDF_AVAILABLE:
        return "Error: pypdf is not installed. Run: pip install pypdf"
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"--- Page {i+1} ---\n{page_text.strip()}")
        if not pages:
            return "Error: PDF appears to have no extractable text (may be image-only/scanned)."
        full_text = "\n\n".join(pages)
        return _clean_text_content(full_text)
    except Exception as e:
        return f"Error extracting PDF text: {e}"


def _fetch_via_playwright(url: str) -> str:
    """Fallback: use a headless browser to render JS-heavy pages."""
    if not _PLAYWRIGHT_AVAILABLE:
        return ""
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            # Extract visible text from body after JS execution
            text = page.evaluate("() => document.body.innerText")
            browser.close()
        return _clean_text_content(text or "")
    except Exception as e:
        console.print(f"[dim]Playwright fallback failed: {e}[/dim]")
        return ""


def _extract_youtube_video_id(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""
        if "youtu.be" in host:
            return path.strip("/").split("/")[0]
        if "youtube.com" in host:
            if path == "/watch":
                return (parse_qs(parsed.query).get("v") or [""])[0]
            if path.startswith("/shorts/") or path.startswith("/embed/"):
                parts = [p for p in path.split("/") if p]
                if len(parts) >= 2:
                    return parts[1]
    except Exception:
        return ""
    return ""


def _extract_youtube_description(page_html: str) -> str:
    patterns = [
        r'"shortDescription":"(.*?)","isCrawlable"',
        r'"shortDescription":"(.*?)","allowRatings"',
    ]
    raw = ""
    for pattern in patterns:
        m = re.search(pattern, page_html, flags=re.DOTALL)
        if m:
            raw = m.group(1)
            break
    if not raw:
        return ""
    try:
        return json.loads(f"\"{raw}\"").strip()
    except Exception:
        return raw.replace("\\n", "\n").replace("\\\"", "\"").strip()


def _extract_youtube_title(page_html: str) -> str:
    patterns = [
        r'"title":"(.*?)","lengthSeconds"',
        r'"videoDetails":\{"videoId":"[^"]+","title":"(.*?)"',
    ]
    raw = ""
    for pattern in patterns:
        m = re.search(pattern, page_html, flags=re.DOTALL)
        if m:
            raw = m.group(1)
            break
    if not raw:
        return ""
    try:
        return json.loads(f"\"{raw}\"").strip()
    except Exception:
        return raw.replace("\\n", " ").replace("\\\"", "\"").strip()


def _youtube_lang_label(lang_code: str) -> str:
    lang = (lang_code or "").lower()
    if lang.startswith("en"):
        return "English"
    if lang.startswith("zh"):
        return "Chinese"
    return lang_code or "Unknown"


def _youtube_lang_rank(lang_code: str) -> int:
    lang = (lang_code or "").lower()
    if lang.startswith("en"):
        return 0
    if lang.startswith("zh"):
        return 1
    return 2


def _extract_youtube_transcript(video_id: str, headers: Dict[str, str]) -> Tuple[str, str]:
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            list_resp = client.get(
                "https://www.youtube.com/api/timedtext",
                params={"type": "list", "v": video_id},
                headers=headers,
            )
        if list_resp.status_code >= 400 or not list_resp.text.strip():
            return "", ""
        soup = BeautifulSoup(list_resp.text, "xml")
        tracks = soup.find_all("track")
        if not tracks:
            return "", ""
        candidates = []
        for track in tracks:
            lang_code = (track.get("lang_code") or "").strip().lower()
            candidates.append({
                "lang": track.get("lang_code") or "en",
                "name": track.get("name") or "",
                "kind": track.get("kind") or "",
                "rank": _youtube_lang_rank(lang_code),
            })
        if not candidates:
            return "", ""
        candidates.sort(key=lambda t: (t["rank"], 1 if t["kind"] == "asr" else 0, t["lang"]))
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            for c in candidates:
                params = {"v": video_id, "lang": c["lang"]}
                if c["name"]:
                    params["name"] = c["name"]
                if c["kind"]:
                    params["kind"] = c["kind"]
                transcript_resp = client.get(
                    "https://www.youtube.com/api/timedtext",
                    params={**params, "fmt": "vtt"},
                    headers=headers,
                )
                if transcript_resp.status_code < 400 and transcript_resp.text.strip():
                    text = _parse_vtt_transcript(transcript_resp.text)
                    if text:
                        return text, _youtube_lang_label(str(c["lang"]))
                transcript_resp = client.get(
                    "https://www.youtube.com/api/timedtext",
                    params=params,
                    headers=headers,
                )
                if transcript_resp.status_code >= 400 or not transcript_resp.text.strip():
                    continue
                text = _extract_text_from_xml_transcript(transcript_resp.text)
                if text:
                    return text, _youtube_lang_label(str(c["lang"]))
    except Exception:
        return "", ""
    return "", ""


def _extract_youtube_transcript_from_base_urls(page_html: str, headers: Dict[str, str]) -> Tuple[str, str]:
    base_url_matches = re.findall(r'"baseUrl":"(https:\\/\\/www\\.youtube\\.com\\/api\\/timedtext[^"]+)"', page_html)
    if not base_url_matches:
        return "", ""
    decoded_urls = []
    for raw in base_url_matches:
        try:
            decoded_urls.append(json.loads(f"\"{raw}\""))
        except Exception:
            decoded_urls.append(raw.replace("\\/", "/"))
    seen = set()
    ordered_urls = []
    for u in decoded_urls:
        if u in seen:
            continue
        seen.add(u)
        ordered_urls.append(u)
    candidates = sorted(
        ordered_urls,
        key=lambda u: _youtube_lang_rank((parse_qs(urlparse(u).query).get("lang") or [""])[0].lower()),
    )
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            for u in candidates:
                lang_code = (parse_qs(urlparse(u).query).get("lang") or [""])[0]
                vtt_url = u if "fmt=" in u else f"{u}&fmt=vtt"
                resp = client.get(vtt_url, headers=headers)
                text = ""
                if resp.status_code >= 400 or not resp.text.strip():
                    json3_url = u if "fmt=" in u else f"{u}&fmt=json3"
                    resp = client.get(json3_url, headers=headers)
                    if resp.status_code >= 400 or not resp.text.strip():
                        continue
                    text = _extract_text_from_json3_transcript(resp.text)
                if not text:
                    text = _parse_vtt_transcript(resp.text)
                if text:
                    return text, _youtube_lang_label(lang_code)
    except Exception:
        return "", ""
    return "", ""


def _extract_youtube_transcript_with_ytdlp(video_url: str) -> Tuple[str, str]:
    if not _YTDLP_AVAILABLE:
        return "", ""
    try:
        opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        subtitles = info.get("subtitles") or {}
        auto_subtitles = info.get("automatic_captions") or {}
        all_keys = set(subtitles.keys()) | set(auto_subtitles.keys())
        ordered_lang_keys = sorted(all_keys, key=lambda k: _youtube_lang_rank(str(k).lower()))
        for bucket in (subtitles, auto_subtitles):
            for key in ordered_lang_keys:
                if key not in bucket:
                    continue
                tracks = bucket.get(key) or []
                preferred = []
                fallback = []
                ext_order = {"vtt": 0, "ttml": 1, "srv3": 2, "srv2": 3, "srv1": 4, "json3": 5}
                for track in tracks:
                    ext = str(track.get("ext") or "").lower()
                    if ext in {"json3", "srv3", "srv2", "srv1", "ttml", "vtt"}:
                        preferred.append(track)
                    else:
                        fallback.append(track)
                preferred.sort(key=lambda t: ext_order.get(str(t.get("ext") or "").lower(), 999))
                for track in preferred + fallback:
                    turl = str(track.get("url") or "").strip()
                    if not turl:
                        continue
                    try:
                        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                            resp = client.get(turl)
                        if resp.status_code >= 400 or not resp.text.strip():
                            continue
                        ext = str(track.get("ext") or "").lower()
                        if ext == "vtt":
                            text = _parse_vtt_transcript(resp.text)
                        elif "json3" in ext:
                            text = _extract_text_from_json3_transcript(resp.text)
                        else:
                            text = _extract_text_from_xml_transcript(resp.text)
                        if text:
                            return text, _youtube_lang_label(str(key))
                    except Exception:
                        continue
    except Exception:
        return "", ""
    return "", ""


def _preserve_short_lines(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if len(value) > _MAX_CONTENT_CHARS:
        value = value[:_MAX_CONTENT_CHARS] + "\n\n... [Content Truncated] ..."
    return value


def _parse_vtt_transcript(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)
    cues: List[Dict[str, Any]] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith("WEBVTT"):
            continue
        if lines[0].startswith("Kind:") or lines[0].startswith("Language:"):
            continue
        timing_idx = 0
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            timing_idx = 1
        if "-->" not in lines[timing_idx]:
            continue
        start_ms, end_ms = _parse_vtt_timing_line(lines[timing_idx])
        payload_lines = lines[timing_idx + 1:]
        if not payload_lines:
            continue
        payload = _clean_caption_text(" ".join(payload_lines))
        if not payload:
            continue
        cues.append({"start_ms": start_ms, "end_ms": end_ms, "text": payload})
    return _merge_timed_cues(cues)


def _extract_text_from_xml_transcript(raw_xml: str) -> str:
    tsoup = BeautifulSoup(raw_xml, "xml")
    cues: List[Dict[str, Any]] = []
    for node in tsoup.find_all("text"):
        payload = _clean_caption_text(html.unescape((node.get_text() or "").strip()))
        if not payload:
            continue
        try:
            start_s = float(str(node.get("start") or "0"))
        except Exception:
            start_s = 0.0
        try:
            dur_s = float(str(node.get("dur") or "0"))
        except Exception:
            dur_s = 0.0
        cues.append({
            "start_ms": int(start_s * 1000),
            "end_ms": int((start_s + max(dur_s, 0.0)) * 1000),
            "text": payload,
        })
    return _merge_timed_cues(cues)


def _extract_text_from_json3_transcript(raw_json: str) -> str:
    try:
        payload = json.loads(raw_json)
    except Exception:
        return ""
    events = payload.get("events") or []
    cues: List[Dict[str, Any]] = []
    for event in events:
        joined = " ".join(str(seg.get("utf8") or "") for seg in (event.get("segs") or []))
        value = _clean_caption_text(joined.replace("\n", " ").strip())
        if not value:
            continue
        start_ms = int(event.get("tStartMs") or 0)
        dur_ms = int(event.get("dDurationMs") or 0)
        cues.append({
            "start_ms": start_ms,
            "end_ms": start_ms + max(dur_ms, 0),
            "text": value,
        })
    return _merge_timed_cues(cues)


def _normalize_transcript_text(text: str) -> str:
    value = html.unescape(text or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\u200b", " ").replace("&nbsp;", " ")
    value = value.replace(">>>", " ").replace(">>", " ").replace(">", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.lower()
    return value


def _clean_caption_text(text: str) -> str:
    value = html.unescape(text or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\u200b", " ").replace("&nbsp;", " ")
    value = value.replace(">>>", " ").replace(">>", " ").replace(">", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""
    return value


def _parse_vtt_timestamp_to_ms(value: str) -> int:
    raw = value.strip().split(" ")[0]
    m = re.match(r"^(?:(\d+):)?(\d+):(\d+)(?:\.(\d+))?$", raw)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    minute = int(m.group(2) or 0)
    sec = int(m.group(3) or 0)
    frac = (m.group(4) or "0").ljust(3, "0")[:3]
    ms = int(frac)
    return ((h * 60 + minute) * 60 + sec) * 1000 + ms


def _parse_vtt_timing_line(line: str) -> Tuple[int, int]:
    parts = line.split("-->")
    if len(parts) != 2:
        return 0, 0
    start = _parse_vtt_timestamp_to_ms(parts[0].strip())
    end = _parse_vtt_timestamp_to_ms(parts[1].strip())
    return start, end


def _max_token_overlap(prev_tokens: List[str], current_tokens: List[str], max_window: int = 24) -> int:
    max_k = min(len(prev_tokens), len(current_tokens), max_window)
    for k in range(max_k, 0, -1):
        if prev_tokens[-k:] == current_tokens[:k]:
            return k
    return 0


def _merge_timed_cues(cues: List[Dict[str, Any]]) -> str:
    if not cues:
        return ""
    ordered = sorted(cues, key=lambda c: (int(c.get("start_ms") or 0), int(c.get("end_ms") or 0)))
    out_tokens: List[str] = []
    last_end_ms = -1
    for cue in ordered:
        text = _clean_caption_text(str(cue.get("text") or ""))
        if not text:
            continue
        start_ms = int(cue.get("start_ms") or 0)
        end_ms = int(cue.get("end_ms") or 0)
        curr_tokens = text.split()
        if not curr_tokens:
            continue
        overlap = _max_token_overlap(out_tokens, curr_tokens)
        if overlap > 0:
            curr_tokens = curr_tokens[overlap:]
        elif start_ms <= last_end_ms + 300 and len(out_tokens) >= 10:
            tail_text = " ".join(out_tokens[-10:]).lower()
            curr_text = " ".join(curr_tokens).lower()
            if curr_text in tail_text:
                continue
        if curr_tokens:
            out_tokens.extend(curr_tokens)
        last_end_ms = max(last_end_ms, end_ms)
    return _normalize_transcript_text(" ".join(out_tokens))


def _fetch_youtube_content(url: str, headers: Dict[str, str]) -> str:
    video_id = _extract_youtube_video_id(url)
    if not video_id:
        return "Error: Could not parse YouTube video ID."
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    title = ""
    description = ""
    page_html = ""
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            oembed = client.get(
                "https://www.youtube.com/oembed",
                params={"url": watch_url, "format": "json"},
                headers=headers,
            )
            if oembed.status_code < 400:
                obj = oembed.json()
                title = str(obj.get("title") or "").strip()
            page_resp = client.get(watch_url, headers=headers)
            if page_resp.status_code < 400:
                page_html = page_resp.text
                description = _extract_youtube_description(page_html)
                if not title:
                    title = _extract_youtube_title(page_html)
        if (not title or not description) and _YTDLP_AVAILABLE:
            opts = {
                "skip_download": True,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(watch_url, download=False)
            if not title:
                title = str(info.get("title") or "").strip()
            if not description:
                description = str(info.get("description") or "").strip()
    except Exception:
        pass
    transcript = ""
    transcript_label = ""
    transcript, transcript_label = _extract_youtube_transcript(video_id, headers)
    if not transcript and page_html:
        transcript, transcript_label = _extract_youtube_transcript_from_base_urls(page_html, headers)
    if not transcript:
        transcript, transcript_label = _extract_youtube_transcript_with_ytdlp(watch_url)
    transcript_heading = f"Transcript ({transcript_label}):" if transcript_label else "Transcript (English/Chinese):"
    sections = [
        f"YouTube URL: {watch_url}",
        f"Title: {title or '[Unavailable]'}",
        "",
        "Description:",
        description or "[Unavailable]",
        "",
        transcript_heading,
        transcript or "[Unavailable]",
    ]
    return _preserve_short_lines("\n".join(sections))


def fetch_and_parse_url(url: str) -> str:
    """
    Fetches a URL and returns clean text content.
    Handles: HTML pages, PDFs, plain text files, and YouTube videos.
    Falls back to Playwright for JS-heavy pages with thin content.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            local_path = Path(parsed.path or "").expanduser()
            if not local_path.exists() or not local_path.is_file():
                return f"Local file not found: {local_path}"
            text = local_path.read_text(encoding="utf-8", errors="replace")
            return _clean_text_content(text)
        if _extract_youtube_video_id(url):
            console.print("[dim]Detected YouTube URL, extracting description and transcript...[/dim]")
            return _fetch_youtube_content(url, headers)

        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            raw_bytes = response.content

        # ── Layer 1: File-type detection ──────────────────────────────────
        url_path = url.split("?")[0].lower()
        is_pdf = (
            url_path.endswith(".pdf")
            or "application/pdf" in content_type
            or "pdf" in content_type
        )
        is_plaintext = any(url_path.endswith(ext) for ext in (".txt", ".md", ".csv", ".log", ".rst"))

        if is_pdf:
            console.print(f"[dim]Detected PDF, extracting text via pypdf...[/dim]")
            return _extract_pdf_text(raw_bytes)

        if is_plaintext:
            console.print(f"[dim]Detected plain text file, decoding...[/dim]")
            text = raw_bytes.decode("utf-8", errors="replace")
            return _clean_text_content(text)

        # ── Layer 2: HTML parsing ─────────────────────────────────────────
        try:
            html_content = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            html_content = response.text

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove noisy elements
        for tag in _NOISE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()
        # Also remove elements with cookie/ad-related classes or IDs
        for el in soup.find_all(True, attrs={"class": re.compile(r'cookie|consent|banner|popup|overlay|ad-|ads-', re.I)}):
            el.decompose()
        for el in soup.find_all(True, attrs={"id": re.compile(r'cookie|consent|banner|popup|overlay', re.I)}):
            el.decompose()

        # Find the richest content container
        main_content = (
            soup.find('article')
            or soup.find('main')
            or soup.find(id=re.compile(r'content|main|body|post', re.I))
            or soup.find(class_=re.compile(r'content|main|article|post|entry', re.I))
            or soup.body
        )

        if main_content:
            md_text = markdownify.markdownify(
                str(main_content),
                heading_style="ATX",
                strip=['img', 'a'],
            )
        else:
            # Last-resort: raw text extraction from soup
            md_text = soup.get_text(separator='\n', strip=True)

        md_text = _clean_text_content(md_text)

        # ── Layer 3: Playwright fallback for thin/empty content ───────────
        if len(md_text) < _PLAYWRIGHT_THRESHOLD and _PLAYWRIGHT_AVAILABLE:
            console.print(f"[dim]Content too thin ({len(md_text)} chars), trying Playwright fallback...[/dim]")
            playwright_text = _fetch_via_playwright(url)
            if len(playwright_text) > len(md_text):
                console.print(f"[dim]Playwright returned {len(playwright_text)} chars, using that.[/dim]")
                return playwright_text

        if not md_text:
            return "Error: Could not extract any meaningful content from this page."

        return md_text

    except httpx.HTTPStatusError as e:
        # For PDFs behind auth walls, try playwright
        if _PLAYWRIGHT_AVAILABLE:
            console.print(f"[dim]HTTP {e.response.status_code} error, trying Playwright...[/dim]")
            result = _fetch_via_playwright(url)
            if result:
                return result
        return f"HTTP Error {e.response.status_code} while fetching URL: {url}"
    except httpx.HTTPError as e:
        return f"Network error while fetching URL: {str(e)}"
    except Exception as e:
        return f"Failed to parse URL: {str(e)}\n{traceback.format_exc(limit=3)}"
    
# ==========================================
# 1. Action Parsers
# ==========================================

def parse_text_actions(content: str, allowlist: List[str]) -> List[AgentAction]:
    """
    Fallback text parser to convert plain text markdown into AgentAction protocol.
    Optimized for robustness.
    """
    actions = []

    # ── Strip <think>...</think> reasoning blocks before parsing tool tags ───────
    # Reasoning models write tool calls in <think> as exploratory drafts; the real
    # call appears AFTER </think>.  We keep original `content` for WRITE_FILE /
    # diff extraction (those patterns appear in the answer section too).
    think_stripped = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    parse_content = think_stripped if think_stripped.strip() else content

    # ── Parser Priority Order ───────────────────────────────────────────────────
    # (All applied against think_stripped `parse_content` so reasoning drafts
    #  inside <think> blocks are invisible to the parser.)

    # ① CANONICAL XML child-tag format — what the model's training produces:
    #    <tool_call><write_file><path>./foo.py</path><content>...</content></write_file></tool_call>
    for match in re.finditer(
        r'(?:<tool_call>\s*)?<write_file>\s*<path>\s*(.*?)\s*</path>\s*<content>(.*?)</content>\s*</write_file>(?:\s*</tool_call>)?',
        parse_content, re.DOTALL | re.IGNORECASE
    ):
        fpath = match.group(1).strip()
        file_content = match.group(2)
        target_path = resolve_path(fpath, allowlist)
        if target_path and len(file_content.strip()) >= 5:
            actions.append(ActionWriteFile(path=str(target_path), content=file_content.rstrip() + "\n"))

    # ① search_and_replace canonical XML:
    #    <tool_call><search_and_replace><path>...</path><old_text>...</old_text><new_text>...</new_text></search_and_replace></tool_call>
    for match in re.finditer(
        r'(?:<tool_call>\s*)?<search_and_replace>\s*<path>\s*(.*?)\s*</path>\s*<old_text>(.*?)</old_text>\s*<new_text>(.*?)</new_text>\s*</search_and_replace>(?:\s*</tool_call>)?',
        parse_content, re.DOTALL | re.IGNORECASE
    ):
        fpath   = match.group(1).strip()
        old_txt = match.group(2)
        new_txt = match.group(3)
        target_path = resolve_path(fpath, allowlist)
        if target_path:
            actions.append(ActionReplaceText(path=str(target_path), old_text=old_txt, new_text=new_txt))

    # ② Fallback: old key=value format inside <write_file> tag (pre-standardisation outputs)
    #    <tool_call><write_file>path=./foo.py\ncontent=...</write_file></tool_call>
    if not actions:
        for match in re.finditer(
            r'(?:<tool_call>\s*)?<write_file>\s*path\s*=\s*(\S+)\s*\ncontent=(.*?)</write_file>(?:\s*</tool_call>)?',
            parse_content, re.DOTALL | re.IGNORECASE
        ):
            fpath = match.group(1).strip()
            file_content = match.group(2).rstrip()
            target_path = resolve_path(fpath, allowlist)
            if target_path and len(file_content.strip()) >= 5:
                actions.append(ActionWriteFile(path=str(target_path), content=file_content + "\n"))

    # ③ Fallback: WRITE_FILE with code-fence body
    #    WRITE_FILE: ./foo.py\n```python\n...\n```
    if not actions:
        for match in re.finditer(
            r'WRITE_FILE:\s*(\S+)\s*\n```(?:\w+)?\n(.*?)```',
            parse_content, re.DOTALL
        ):
            fpath = match.group(1).strip()
            file_content = match.group(2).rstrip()
            target_path = resolve_path(fpath, allowlist)
            if target_path and len(file_content.strip()) >= 5:
                actions.append(ActionWriteFile(path=str(target_path), content=file_content + "\n"))

    # ④ Legacy WRITE_FILE (Format B — <<<CONTENT ... CONTENT>>>)
    write_actions = extract_write_file_actions_v2(content)
    if write_actions:
        for path, text in write_actions:
            target_path = resolve_path(path, allowlist)
            if target_path:
                actions.append(ActionWriteFile(path=str(target_path), content=text))

    # ⑤ Unified Diff (Format A)
    diff = extract_all_diffs(content)
    if diff:
        actions.append(ActionApplyDiff(diff_text=diff))
        
    # ── Universal XML Tool Parser (Format C) ──────────────────────────────────
    # Extracts ANY structure matching <tool_call><[TOOL_NAME]>[ARGS]</[TOOL_NAME]></tool_call>
    # and blindly parses its interior `<[ARG_NAME]>val</[ARG_NAME]>` children into a dictionary.
    
    # 1. Isolate the internal tool tag block
    for match in re.finditer(r'(?:<tool_call>\s*)?<([a-zA-Z0-9_]+)>(.*?)</\1>(?:\s*</tool_call>)?', parse_content, re.DOTALL):
        tool_name = match.group(1).strip()
        inner_content = match.group(2).strip()
        
        # Don't capture known parser-specific tags as native tools
        if tool_name in ["write_file", "search_and_replace"]:
            continue
            
        args_dict = {}
        
        # 2. Extract inner specific arguments if they are tagged (e.g., <query>apple</query>)
        has_child_tags = False
        for arg_match in re.finditer(r'<([a-zA-Z0-9_]+)>(.*?)</\1>', inner_content, re.DOTALL):
            arg_name = arg_match.group(1).strip()
            arg_val = arg_match.group(2).strip()
            args_dict[arg_name] = arg_val
            has_child_tags = True
            
        # 3. Fallback: If it's a single-argument tool like <web_search>apple</web_search> without <query> apples
        if not has_child_tags and inner_content and '<' not in inner_content:
            # We must map this stray string value to the correct argument name defined in the schema
            if tool_name in BASE_TOOLS_SCHEMA and BASE_TOOLS_SCHEMA[tool_name]:
                first_arg_name = list(BASE_TOOLS_SCHEMA[tool_name].keys())[0]
                args_dict[first_arg_name] = inner_content
        
        # Default parameter logic if missing entirely
        if tool_name == "list_directory" and not args_dict:
            args_dict = {"dir_path": "."}
            
        # Only deploy if valid tool
        if tool_name in BASE_TOOLS_SCHEMA or tool_name == "finish_task":
             actions.append(ActionToolCall(name=tool_name, args=args_dict))

    # ── finish_task ────────────────────────────────────────────────────────────
    # Critical: must be detected so text_only agents can exit the ReAct loop.
    # Matches: <tool_call><finish_task>...</finish_task></tool_call>
    #   and:   <tool_call><finish_task/></tool_call>  (self-closing)
    #   and:   <finish_task>...</finish_task>  (bare, without outer tool_call)
    finish_pattern = re.compile(
        r'(?:<tool_call>\s*)?<finish_task(?:>\s*(?:.*?)\s*</finish_task>|/>)(?:\s*</tool_call>)?',
        re.DOTALL | re.IGNORECASE,
    )
    if finish_pattern.search(content):
        # Extract optional summary text inside <summary>...</summary>
        summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""
        actions.append(ActionToolCall(name="finish_task", args={"summary": summary}))

    # 4. Extreme Fallbacks if no explicit format found and we only have 1 target file
    # if not actions and len(allowlist) == 1:
    #     code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
    #     if len(code_blocks) == 1:
    #         block = code_blocks[0].strip()
    #         if "def " in block or "import " in block:
    #             actions.append(ActionWriteFile(path=allowlist[0], content=block))
    #     elif "def " in content or "import " in content:
    #         clean = content.strip()
    #         if clean.startswith("```python"): clean = clean[len("```python"):].strip()
    #         elif clean.startswith("```"): clean = clean[3:].strip()
    #         if clean.endswith("```"): clean = clean[:-3].strip()
    #         actions.append(ActionWriteFile(path=allowlist[0], content=clean))
            
    return actions


# ==========================================
# 2. Advanced Web Search (Domain-Aware)
# ==========================================

def _extract_reference_year(current_time: str) -> Optional[int]:
    if not current_time:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", current_time)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _resolve_youtube_top_k(explicit_top_k: Optional[int] = None) -> int:
    if explicit_top_k is not None:
        try:
            return max(1, min(int(explicit_top_k), 10))
        except Exception:
            return 3
    raw = os.getenv("BATCHAGENT_YOUTUBE_SEARCH_TOP_K", "").strip()
    if not raw:
        return 3
    try:
        return max(1, min(int(raw), 10))
    except Exception:
        return 3


def _parse_youtube_content_fields(content: str) -> Tuple[str, str]:
    title = ""
    description = ""
    title_match = re.search(r"^Title:\s*(.+?)\s*$", content, flags=re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
    desc_match = re.search(r"^Description:\s*([\s\S]*?)^\s*Transcript\s*\(", content, flags=re.MULTILINE)
    if desc_match:
        description = desc_match.group(1).strip()
    return title, description


def _persist_youtube_markdown(video_url: str, parsed_content: str, video_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", video_id or "video")[:24] or "video"
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    md_dir = Path(tempfile.gettempdir()) / "batchagent_youtube_extracts"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = (md_dir / f"youtube_{safe_id}_{timestamp}.md").resolve()
    md_text = "\n".join([
        "# YouTube Extract",
        "",
        f"Source URL: {video_url}",
        "",
        parsed_content.strip(),
        "",
    ])
    md_path.write_text(md_text, encoding="utf-8")
    uploaded_url = ""
    try:
        from BatchAgent.minio_uploader import upload_file
        uploaded_url = upload_file(str(md_path), object_name=f"agent/youtube_extracts/{md_path.name}") or ""
    except Exception:
        uploaded_url = ""
    if uploaded_url:
        return uploaded_url
    return f"file://{md_path}"


def youtube_search_and_extract(query: str, api_key: str, top_k: Optional[int] = None) -> List[Dict[str, str]]:
    if not api_key or api_key == "EMPTY":
        return []
    effective_top_k = _resolve_youtube_top_k(top_k)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    req_headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    video_candidates: List[Dict[str, str]] = []
    try:
        req = urllib.request.Request("https://google.serper.dev/videos", method="POST")
        for k, v in req_headers.items():
            req.add_header(k, v)
        payload = json.dumps({"q": query, "num": max(effective_top_k, 5)}).encode("utf-8")
        with urllib.request.urlopen(req, data=payload, timeout=15) as response:
            body = json.loads(response.read().decode("utf-8"))
        videos = body.get("videos", []) or body.get("organic", []) or []
        for item in videos:
            link = str(item.get("link") or item.get("url") or "").strip()
            if not _extract_youtube_video_id(link):
                continue
            video_candidates.append({
                "url": link,
                "title": str(item.get("title") or "").strip(),
                "snippet": str(item.get("snippet") or item.get("description") or "").strip(),
            })
    except Exception:
        video_candidates = []
    if not video_candidates:
        try:
            req = urllib.request.Request("https://google.serper.dev/search", method="POST")
            for k, v in req_headers.items():
                req.add_header(k, v)
            payload = json.dumps({"q": f"{query} site:youtube.com", "num": max(effective_top_k * 2, 8)}).encode("utf-8")
            with urllib.request.urlopen(req, data=payload, timeout=15) as response:
                body = json.loads(response.read().decode("utf-8"))
            for item in body.get("organic", []):
                link = str(item.get("link") or "").strip()
                if not _extract_youtube_video_id(link):
                    continue
                video_candidates.append({
                    "url": link,
                    "title": str(item.get("title") or "").strip(),
                    "snippet": str(item.get("snippet") or "").strip(),
                })
        except Exception:
            video_candidates = []
    deduped: List[Dict[str, str]] = []
    seen_ids = set()
    for item in video_candidates:
        video_id = _extract_youtube_video_id(item["url"])
        if not video_id or video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        deduped.append({
            "video_id": video_id,
            "url": watch_url,
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
        })
        if len(deduped) >= effective_top_k:
            break
    out: List[Dict[str, str]] = []
    for item in deduped:
        parsed = _fetch_youtube_content(item["url"], headers)
        parsed_title, parsed_desc = _parse_youtube_content_fields(parsed)
        markdown_url = _persist_youtube_markdown(item["url"], parsed, item["video_id"])
        out.append({
            "url": item["url"],
            "title": parsed_title or item.get("title") or item["url"],
            "description": parsed_desc or item.get("snippet") or "",
            "markdown_url": markdown_url,
        })
    return out


def _perform_serper_domain_search(final_query: str, cat: str, api_key: str) -> str:
    url = "https://google.serper.dev/search"
    req = urllib.request.Request(url, method="POST")
    req.add_header("X-API-KEY", api_key)
    req.add_header("Content-Type", "application/json")

    if cat == "general":
        num_results = 8
    elif cat in ("math", "academic", "science", "medical"):
        num_results = 6
    else:
        num_results = 5

    data = json.dumps({"q": final_query, "num": num_results}).encode("utf-8")

    with urllib.request.urlopen(req, data=data, timeout=15) as response:
        res_data = json.loads(response.read().decode("utf-8"))

    organic = res_data.get("organic", [])
    answer_box = res_data.get("answerBox", {})
    knowledge_graph = res_data.get("knowledgeGraph", {})

    results = ["🔎 [Source: Serper Web Search]"]

    if answer_box and "snippet" in answer_box:
        results.append(f"⭐ [Direct Answer]: {answer_box['snippet']}\n")

    if knowledge_graph and "description" in knowledge_graph:
        results.append(f"📚 [Knowledge Panel]: {knowledge_graph['description']}\n")

    for i, item in enumerate(organic):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        date = item.get("date", "")
        date_str = f" ({date})" if date else ""
        results.append(f"[{i+1}] {title}{date_str}\n{snippet}\nURL: {link}\n")

    return "\n".join(results) if len(results) > 1 else f"🔎 [Source: Serper Web Search]\nNo results found for '{final_query}'."


def wikimedia_search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    limit = max(1, min(int(top_k), 10))
    req = urllib.request.Request(
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={quote(query)}&srlimit={limit}&format=json&utf8=1",
        method="GET",
    )
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    items = payload.get("query", {}).get("search", []) or []
    results: List[Dict[str, str]] = []
    for item in items:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        snippet_html = str(item.get("snippet") or "").strip()
        description = html.unescape(re.sub(r"<[^>]+>", "", snippet_html)).strip()
        wiki_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        results.append({
            "title": title,
            "description": description,
            "url": wiki_url,
            "source": "Wikimedia/MediaWiki",
        })
    return results


def _format_youtube_search_results(items: List[Dict[str, str]]) -> str:
    if not items:
        return ""
    blocks = ["🎬 [Source: YouTube Search + YouTube Parser]"]
    for i, item in enumerate(items):
        blocks.append(
            f"[YT-{i+1}] {item.get('title', '').strip()}\n"
            f"{item.get('description', '').strip()}\n"
            f"URL: {item.get('url', '').strip()}\n"
            f"Markdown URL: {item.get('markdown_url', '').strip()}\n"
        )
    return "\n".join(blocks)


def _format_wikimedia_search_results(items: List[Dict[str, str]]) -> str:
    if not items:
        return ""
    blocks = ["📚 [Source: Wikimedia / MediaWiki API]"]
    for i, item in enumerate(items):
        blocks.append(
            f"[WM-{i+1}] {item.get('title', '').strip()}\n"
            f"{item.get('description', '').strip()}\n"
            f"URL: {item.get('url', '').strip()}\n"
            f"Source: {item.get('source', 'Wikimedia/MediaWiki').strip()}\n"
        )
    return "\n".join(blocks)


def _perform_document_rag_search(query: str, top_k: int = 3) -> str:
    """Search loaded document RAG index (if any). Returns empty string if no document is loaded."""
    try:
        from BatchAgent.document_agent.document_agent_v1 import get_active_chunk_index
        chunk_index = get_active_chunk_index()
        if chunk_index is None:
            return ""
        return chunk_index.search_formatted(query, top_k=top_k)
    except Exception as e:
        console.print(f"[dim]Document RAG search skipped: {e}[/dim]")
        return ""


def perform_domain_aware_search(query: str, category: str, api_key: str, current_time: str = "") -> str:
    if not api_key or api_key == "EMPTY":
        return "System Error: SERPER_API_KEY is not configured."

    # ── Domain filter map (covers all DOMAIN_REGISTRY keys + common aliases) ──
    # Filters are deliberately broad — multiple high-quality sources, not single-site
    domain_filters: Dict[str, str] = {
        # Core DOMAIN_REGISTRY keys
        "news":         "site:reuters.com OR site:apnews.com OR site:bbc.com OR site:bloomberg.com OR site:theguardian.com",
        "academic":     "site:arxiv.org OR site:scholar.google.com OR site:semanticscholar.org OR site:pubmed.ncbi.nlm.nih.gov OR site:researchgate.net",
        "medical":      "site:pubmed.ncbi.nlm.nih.gov OR site:medlineplus.gov OR site:nih.gov OR site:mayoclinic.org OR site:webmd.com",
        "software_eng": "site:stackoverflow.com OR site:github.com OR site:dev.to OR site:docs.python.org OR site:pypi.org",
        "math":         "site:math.stackexchange.com OR site:artofproblemsolving.com OR site:mathworld.wolfram.com OR site:khanacademy.org OR site:brilliant.org",
        "science":      "site:nature.com OR site:sciencedirect.com OR site:phys.org OR site:science.org OR site:wolframalpha.com",
        "language":     "site:en.wiktionary.org OR site:languageguide.org OR site:bbc.co.uk/languages OR site:italki.com",
        "business":     "site:sec.gov OR site:finance.yahoo.com OR site:bloomberg.com OR site:investopedia.com OR site:marketwatch.com",
        "assistant":    "site:superuser.com OR site:askubuntu.com OR site:serverfault.com OR site:apple.stackexchange.com",
        "sales_support":"site:zendesk.com OR site:hubspot.com OR site:salesforce.com OR site:freshdesk.com",
        # Common aliases
        "code":         "site:github.com OR site:stackoverflow.com OR site:docs.python.org OR site:pypi.org OR site:realpython.com",
        "finance":      "site:sec.gov OR site:finance.yahoo.com OR site:bloomberg.com OR site:investopedia.com",
        "health":       "site:pubmed.ncbi.nlm.nih.gov OR site:nih.gov OR site:mayoclinic.org OR site:webmd.com",
        "programming":  "site:stackoverflow.com OR site:github.com OR site:realpython.com OR site:docs.python.org",
        "research":     "site:arxiv.org OR site:semanticscholar.org OR site:scholar.google.com OR site:jstor.org",
        # Default fallback
        "general":      "",
    }

    # ── Category alias normalisation ──────────────────────────────────────────
    _aliases: Dict[str, str] = {
        "software":            "software_eng",
        "software_engineering":"software_eng",
        "engineering":         "software_eng",
        "medicine":            "medical",
        "biology":             "medical",
        "physics":             "science",
        "chemistry":           "science",
        "mathematics":         "math",
        "maths":               "math",
        "statistics":          "math",
        "economics":           "business",
        "stock":               "business",
        "stocks":              "business",
        "investment":          "business",
        "support":             "sales_support",
        "crm":                 "sales_support",
        "system":              "assistant",
        "computer":            "assistant",
        "paper":               "academic",
        "papers":              "academic",
        "python":              "code",
        "javascript":          "code",
        "js":                  "code",
    }

    cat = category.lower().strip()
    cat = _aliases.get(cat, cat)  # resolve alias first
    if cat not in domain_filters:
        cat = "general"  # safe fallback

    filter_str = domain_filters[cat]
    # Only append site filter when non-empty (avoids polluting general queries)
    reference_year = _extract_reference_year(current_time)
    normalized_query = query.strip()
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", normalized_query))
    has_relative_time = bool(re.search(r"\b(latest|newest|current|today|this year|recent)\b", normalized_query, flags=re.IGNORECASE))
    if reference_year and not has_year and has_relative_time:
        normalized_query = f"{normalized_query} in {reference_year}"
    if reference_year and not has_year and not has_relative_time:
        normalized_query = f"{normalized_query} as of {reference_year}"
    final_query = f"{normalized_query} {filter_str}".strip() if filter_str else normalized_query

    console.print(f"[dim]Routing search -> Category: '{cat}', Final Query: '{final_query}'[/dim]")

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            web_future = executor.submit(_perform_serper_domain_search, final_query, cat, api_key)
            yt_future = executor.submit(youtube_search_and_extract, normalized_query, api_key, None)
            wm_future = executor.submit(wikimedia_search, normalized_query, _resolve_youtube_top_k(None))
            doc_future = executor.submit(_perform_document_rag_search, normalized_query)
            web_output = web_future.result()
            youtube_items = yt_future.result()
            wikimedia_items = wm_future.result()
            doc_output = doc_future.result()
        youtube_output = _format_youtube_search_results(youtube_items)
        wikimedia_output = _format_wikimedia_search_results(wikimedia_items)
        outputs = [web_output]
        if doc_output:
            outputs.append(doc_output)
        if youtube_output:
            outputs.append(youtube_output)
        if wikimedia_output:
            outputs.append(wikimedia_output)
        if outputs:
            return "\n\n".join(outputs)
        return web_output
    except urllib.error.URLError as e:
        return f"Network Error during search: {str(e)}"
    except Exception as e:
        return f"Search processing failed: {str(e)}"


# ==========================================
# 3. Universal Tool Handler Class
# ==========================================

class UniversalToolHandler:
    """
    Central dispatcher that takes AgentActions (parsed from JSON or Text)
    and executes them in the local environment or via external APIs.
    """
    def __init__(self, config, turn_dir: Path, allowlist: List[str], dynamic_tools_mapping: dict = None):
        self.config = config
        self.turn_dir = turn_dir
        self.allowlist = allowlist
        self.dynamic_tools_mapping = dynamic_tools_mapping or {}
        self.written_files: list = []
        self.written_file_records: list = []

    def _build_written_file_record(self, file_path: Path) -> Dict[str, Any]:
        ext = file_path.suffix.lstrip(".").lower()
        size = file_path.stat().st_size if file_path.exists() else 0
        content = ""
        if ext in {"md", "markdown", "txt"} or size < 200 * 1024:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = ""
        url = ""
        try:
            from BatchAgent.minio_uploader import upload_file
            session_name = getattr(self.config, "session_dir", Path(".")).name
            object_name = f"agent/{session_name}/{file_path.name}"
            url = upload_file(str(file_path), object_name=object_name) or ""
        except Exception:
            url = ""
        return {
            "name": file_path.name,
            "ext": ext,
            "size": size,
            "url": url,
            "local_path": str(file_path),
            "content": content,
        }

    def _execute_code_mutations(self, actions: List[AgentAction], full_content: str) -> bool:
        """
        Executes file modification actions. 
        Returns True if any file was successfully changed.
        """
        changes_applied = False
        
        for action in actions:
            if isinstance(action, ActionApplyDiff):
                diff = action.diff_text
                (self.turn_dir / "patch.diff").write_text(diff, encoding="utf-8")
                
                # Strategy 1: Strict Git Apply
                if is_git_repo():
                    applied = apply_patch_guarded(diff, self.turn_dir, auto_approve=self.config.auto_approve)
                    if applied:
                        changes_applied = True
                        console.print("[green]Strict diff applied successfully.[/green]")
                        continue
                
                # Strategy 2: Fuzzy Patch
                file_diffs = re.split(r'(?=^diff --git )', diff, flags=re.MULTILINE)
                fuzzy_successes, fuzzy_total = 0, 0
                fuzzy_logs = ["\n--- Fuzzy Patch Attempt ---"]
                
                for fd in file_diffs:
                    if not fd.strip().startswith("diff --git"): continue
                    fuzzy_total += 1
                    match = re.search(r'diff --git a/\S+ b/(\S+)', fd)
                    if match:
                        raw_path = match.group(1)
                        fuzzy_logs.append(f"Processing diff for: {raw_path}")
                        target_path = resolve_path(raw_path, self.allowlist)
                        if target_path:
                            if apply_fuzzy_patch(target_path, fd, log_buffer=fuzzy_logs):
                                fuzzy_successes += 1
                                fuzzy_logs.append(">> Success")
                            else:
                                fuzzy_logs.append(">> Failed")
                        else:
                            fuzzy_logs.append(f"[red]Skipping diff for unresolved path: {raw_path}[/red]")
                
                try:
                    with open(self.turn_dir / "apply.log", "a", encoding="utf-8") as f:
                        f.write("\n".join(fuzzy_logs) + "\n")
                except Exception as e:
                    console.print(f"Failed to append to apply.log: {e}")

                if fuzzy_successes > 0:
                    changes_applied = True
                    console.print(f"[green]Fuzzy patch applied ({fuzzy_successes}/{fuzzy_total} files).[/green]")
                else:
                    console.print(f"[red]Patch failed ({fuzzy_total} hunks). See patch.diff and apply.log.[/red]")
                    
            elif isinstance(action, ActionWriteFile):
                target_path = resolve_path(action.path, self.allowlist)
                if target_path:
                    if apply_write_files([(str(target_path), action.content)], self.allowlist, self.turn_dir):
                        changes_applied = True
                        self.written_files.append(str(target_path))
                        self.written_file_records.append(self._build_written_file_record(target_path))
                else:
                    console.print(f"[red]Skipping WRITE_FILE for unresolved path: {action.path}[/red]")
                    
            elif isinstance(action, ActionReplaceText):
                target_path = resolve_path(action.path, self.allowlist)
                if target_path and target_path.exists():
                    file_text = target_path.read_text(encoding="utf-8")
                    if action.old_text in file_text:
                        new_text = file_text.replace(action.old_text, action.new_text, 1)
                        target_path.write_text(new_text, encoding="utf-8")
                        console.print(f"[green]Replaced text in {target_path}[/green]")
                        changes_applied = True
                    else:
                        console.print(f"[red]search_and_replace failed: 'old_text' not found in {target_path}[/red]")
                else:
                    console.print(f"[red]search_and_replace skipped: unresolved or missing file {action.path}[/red]")

        # Fallback: Check for extractable new files if diff methods failed entirely
        if not changes_applied and full_content and extract_all_diffs(full_content):
            console.print("[yellow]All patch methods failed. Checking for extractable new files in diff...[/yellow]")
            diff_files = extract_files_from_diff(extract_all_diffs(full_content))
            if diff_files and apply_write_files(diff_files, self.allowlist, self.turn_dir):
                changes_applied = True
                self.written_files.extend([p for p, _ in diff_files])
                for p, _ in diff_files:
                    resolved = resolve_path(p, self.allowlist)
                    if resolved and resolved.exists():
                        self.written_file_records.append(self._build_written_file_record(resolved))
                console.print("[green]Wrote new files extracted from diff.[/green]")

        return changes_applied

    def execute(self, actions: List[AgentAction], full_llm_content: str) -> Tuple[bool, str]:
        """
        Executes all actions.
        Returns: (has_mutation_occurred: bool, observation_string: str)
        """
        tool_results = []
        has_mutation = False
        mutation_actions = []

        for action in actions:
            try:
                # -----------------------------------------
                # 1. Read / Search Tools (The "Eyes")
                # -----------------------------------------
                if isinstance(action, ActionToolCall):
                    res = ""
                    name = action.name
                    args = action.args
                    
                    # === [防洪补丁]：终端显示截断 ===
                    # 遍历 args，把超过 80 字符的超长字符串截断，保护终端 UI
                    safe_args = {}
                    for k, v in args.items():
                        val_str = str(v)
                        safe_args[k] = val_str[:80] + "... [TRUNCATED]" if len(val_str) > 80 else v
                    
                    console.print(f"[bold magenta]🛠️ Tool Execution:[/bold magenta] {name}({safe_args})")
                    
                    # === [防线补丁 1]：拦截捏造的 XML 写入工具 ===
                    if name.upper() in ["WRITE_FILE", "WRITE", "CREATE_FILE"]:
                        res = "System Guardrail: Do NOT use XML tool calls for writing files. You MUST use the pure Markdown Format B (`WRITE_FILE: filepath\\n<<<CONTENT...`) outside of any XML tags."
                        tool_results.append(f"### Result for {name}\n```text\n{res}\n```")
                        continue

                    # --- 正常工具路由 ---
                    if name == "web_search":
                        query = args.get("query", "")
                        category = args.get("category", "general")
                        configured_time = getattr(self.config, "current_time", "") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        res = perform_domain_aware_search(query, category, getattr(self.config, 'serper_api_key', ''), configured_time)

                    elif name == "get_document_overview":
                        try:
                            from BatchAgent.document_agent.document_agent_v1 import get_active_section_agent
                            agent = get_active_section_agent()
                            if agent:
                                res = agent.get_overview()
                            else:
                                res = "No document is currently loaded. Load a PDF with the document pipeline first."
                        except Exception as e:
                            res = f"Error getting document overview: {e}"

                    elif name == "read_url":
                        url = args.get("url", "")
                        console.print(f"[dim]Fetching webpage: {url}[/dim]")
                        # 如果没有 fetch_and_parse_url 函数，请确保导入或替换
                        res = fetch_and_parse_url(url) if 'fetch_and_parse_url' in globals() else f"read_url not implemented for {url}"
                        
                    elif name == "search_code":
                        res = search_code(args.get("query", ""))
                        
                    elif name == "find_file":
                        res = find_file(args.get("pattern", ""))
                        
                    elif name == "read_file_chunk":
                        res = read_file_chunk(args.get("filepath", ""), args.get("start_line", 1), args.get("end_line", 1000))
                        
                    elif name == "list_directory":
                        res = list_directory(args.get("dir_path", "."))
                        
                    elif name == "run_bash_command":
                        cmd = args.get("command", "")
                        # === [防线补丁 2]：封杀使用 Bash 写长代码（保护上下文） ===
                        if len(cmd) > 300 and ("cat >" in cmd or "echo " in cmd):
                            res = "System Guardrail: Command too long. Writing files via bash is strictly forbidden. Use Markdown Format B (`WRITE_FILE: ...`) instead."
                        else:
                            res = run_bash_command(cmd)
                    
                    elif name == "json_parse_error":
                        res = args.get("error", "JSON Parse Error")
                    
                    # =========================================================
                    # [NEW] 动态执行自定义工具 (沙盒执行，保障绝对安全)
                    # =========================================================
                    elif name in self.dynamic_tools_mapping:
                        script_path = self.dynamic_tools_mapping[name]
                        target_script = self.config.workspace_dir / script_path
                        
                        if not target_script.exists():
                            res = f"System Error: Cannot find the script '{script_path}'. Did you write the file first?"
                        else:
                            import json
                            # 将大模型传来的 JSON arguments 序列化，作为命令行参数传给脚本
                            args_json = json.dumps(args).replace("'", "'\\''") # 简单的 Bash 单引号转义
                            cmd = f"python3 {script_path} '{args_json}'"
                            
                            # 调用系统的沙盒命令行执行，完美杜绝危险代码直接污染母舰内存
                            code, out = run_shell(cmd, cap=10000, sandbox_container=getattr(self.config, 'sandbox_container', None))
                            
                            if code == 0:
                                res = out.strip()
                            else:
                                res = f"[Custom Tool Execution Error (Exit {code})]:\n{out}"
                                
                    # [NEW] 拦截并执行领域特定工具
                    else:
                        from BatchAgent.domain_tools import DOMAIN_FUNCTIONS
                        if name in DOMAIN_FUNCTIONS:
                            try:
                                func = DOMAIN_FUNCTIONS[name]
                                # 动态解包字典参数并调用函数
                                res = func(**args) 
                            except Exception as e:
                                res = f"Error executing domain tool {name}: {e}"
                        else:
                            res = f"Error: Unknown tool '{name}'"
                        
                    # Format the observation block for the LLM
                    tool_results.append(f"### Result for {name}\n```text\n{res}\n```")

                # -----------------------------------------
                # 2. Write / Patch Tools (The "Hands")
                # -----------------------------------------
                elif isinstance(action, (ActionWriteFile, ActionApplyDiff, ActionReplaceText)):
                    has_mutation = True
                    mutation_actions.append(action)

            except Exception as e:
                error_msg = f"Error executing tool {getattr(action, 'name', 'mutation')}: {str(e)}"
                tool_results.append(error_msg)
                console.print(f"[red]{error_msg}[/red]")

        # Process all code mutations atomically
        if mutation_actions:
            console.print("[cyan]📝 Applying Code Mutations to Disk...[/cyan]")
            success = self._execute_code_mutations(mutation_actions, full_llm_content)
            if success:
                tool_results.append("### System Action\nFile modifications were successfully applied to the disk. Please proceed to verify or conclude.")
            else:
                tool_results.append("### System Error\nFailed to apply file modifications. The diff may be malformed or the file path is incorrect. Please try again using WRITE_FILE format.")

        return has_mutation, "\n\n".join(tool_results)


# --- Helper for text parser ---
# We keep a minimal mock schema here just for text parsing keys
BASE_TOOLS_SCHEMA = {
    "search_code": {"query": ""},
    "find_file": {"pattern": ""},
    "list_directory": {"dir_path": ""},
    "run_bash_command": {"command": ""}
}
