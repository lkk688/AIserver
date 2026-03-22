"""
建议收口这些：
	•	perform_domain_aware_search
	•	fetch_and_parse_url

职责：
	•	所有 web/search/read_url 相关
"""
from __future__ import annotations

import datetime
import html
import io
import json
import os
import re
import tempfile
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import httpx
import markdownify
from bs4 import BeautifulSoup

try:
    from rich.console import Console
    console = Console()
except Exception:
    class _DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = _DummyConsole()

# Optional dependencies
try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

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


# =============================================================================
# Constants
# =============================================================================

_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "svg", "form", "button"
]

_NOISE_PATTERNS = [
    re.compile(r"(cookie|gdpr|privacy|consent|subscribe|newsletter)", re.I),
    re.compile(r"^\s{0,4}[\|\-\–\•→✓✗★☆©®™]{1,3}\s*$"),
]

_MAX_CONTENT_CHARS = 15000
_PLAYWRIGHT_THRESHOLD = 300

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# =============================================================================
# Generic content cleaning
# =============================================================================

def _clean_text_content(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []

    for line in lines:
        stripped = line.strip()

        if len(stripped) < 25 and len(stripped.split()) < 3 and stripped:
            continue

        if any(p.search(stripped) for p in _NOISE_PATTERNS):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) > _MAX_CONTENT_CHARS:
        text = text[:_MAX_CONTENT_CHARS] + "\n\n... [Content Truncated] ..."
    return text


def _preserve_short_lines(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if len(value) > _MAX_CONTENT_CHARS:
        value = value[:_MAX_CONTENT_CHARS] + "\n\n... [Content Truncated] ..."
    return value


# =============================================================================
# PDF extraction
# =============================================================================

def _extract_pdf_text(raw_bytes: bytes) -> str:
    if not _PYPDF_AVAILABLE:
        return "Error: pypdf is not installed. Run: pip install pypdf"

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"--- Page {i + 1} ---\n{page_text.strip()}")

        if not pages:
            return "Error: PDF appears to have no extractable text (may be image-only/scanned)."

        return _clean_text_content("\n\n".join(pages))
    except Exception as e:
        return f"Error extracting PDF text: {e}"


# =============================================================================
# Playwright fallback
# =============================================================================

def _fetch_via_playwright(url: str) -> str:
    if not _PLAYWRIGHT_AVAILABLE:
        return ""

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(user_agent=_DEFAULT_HEADERS["User-Agent"])
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            text = page.evaluate("() => document.body.innerText")
            browser.close()
        return _clean_text_content(text or "")
    except Exception as e:
        console.print(f"[dim]Playwright fallback failed: {e}[/dim]")
        return ""


# =============================================================================
# YouTube helpers
# =============================================================================

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


def _clean_caption_text(text: str) -> str:
    value = html.unescape(text or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\u200b", " ").replace("&nbsp;", " ")
    value = value.replace(">>>", " ").replace(">>", " ").replace(">", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_transcript_text(text: str) -> str:
    value = html.unescape(text or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\u200b", " ").replace("&nbsp;", " ")
    value = value.replace(">>>", " ").replace(">>", " ").replace(">", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.lower()
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

        cues.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": payload,
        })

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
    base_url_matches = re.findall(
        r'"baseUrl":"(https:\\/\\/www\\.youtube\\.com\\/api\\/timedtext[^"]+)"',
        page_html
    )
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


def _parse_youtube_content_fields(content: str) -> Tuple[str, str]:
    title = ""
    description = ""

    title_match = re.search(r"^Title:\s*(.+?)\s*$", content, flags=re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    desc_match = re.search(
        r"^Description:\s*([\s\S]*?)^\s*Transcript\s*\(",
        content,
        flags=re.MULTILINE
    )
    if desc_match:
        description = desc_match.group(1).strip()

    return title, description


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
    headers = dict(_DEFAULT_HEADERS)
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


# =============================================================================
# URL fetch and parse
# =============================================================================

def fetch_and_parse_url(url: str) -> str:
    """
    Fetch a URL and return clean text content.

    Supports:
    - HTML pages
    - PDFs
    - plain text files
    - local file:// text/PDF files
    - YouTube pages
    """
    headers = dict(_DEFAULT_HEADERS)

    try:
        parsed = urlparse(url)

        if parsed.scheme == "file":
            local_path = Path(parsed.path or "").expanduser()
            if not local_path.exists() or not local_path.is_file():
                return f"Local file not found: {local_path}"

            if local_path.suffix.lower() == ".pdf":
                return _extract_pdf_text(local_path.read_bytes())

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

        url_path = url.split("?")[0].lower()
        is_pdf = (
            url_path.endswith(".pdf")
            or "application/pdf" in content_type
            or "pdf" in content_type
        )
        is_plaintext = any(url_path.endswith(ext) for ext in (".txt", ".md", ".csv", ".log", ".rst"))

        if is_pdf:
            console.print("[dim]Detected PDF, extracting text via pypdf...[/dim]")
            return _extract_pdf_text(raw_bytes)

        if is_plaintext:
            console.print("[dim]Detected plain text file, decoding...[/dim]")
            text = raw_bytes.decode("utf-8", errors="replace")
            return _clean_text_content(text)

        try:
            html_content = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")

        for tag in _NOISE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        for el in soup.find_all(True, attrs={"class": re.compile(r"cookie|consent|banner|popup|overlay|ad-|ads-", re.I)}):
            el.decompose()
        for el in soup.find_all(True, attrs={"id": re.compile(r"cookie|consent|banner|popup|overlay", re.I)}):
            el.decompose()

        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id=re.compile(r"content|main|body|post", re.I))
            or soup.find(class_=re.compile(r"content|main|article|post|entry", re.I))
            or soup.body
        )

        if main_content:
            md_text = markdownify.markdownify(
                str(main_content),
                heading_style="ATX",
                strip=["img", "a"],
            )
        else:
            md_text = soup.get_text(separator="\n", strip=True)

        md_text = _clean_text_content(md_text)

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


# =============================================================================
# Search helpers
# =============================================================================

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


def _perform_serper_domain_search(final_query: str, cat: str, api_key: str) -> str:
    url = "https://google.serper.dev/search"
    req = urllib.request.Request(url, method="POST")
    req.add_header("X-API-KEY", api_key)
    req.add_header("Content-Type", "application/json")

    if cat == "general":
        num_results = 10
    elif cat in ("math", "academic", "science", "medical", "research"):
        num_results = 8
    else:
        num_results = 8

    data = json.dumps({"q": final_query, "num": num_results}).encode("utf-8")

    try:
        with urllib.request.urlopen(req, data=data, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return (
                f"[Web Search Unavailable] Serper API rejected the request (HTTP {e.code}). "
                "The SERPER_API_KEY is invalid, expired, or not configured. "
                "Do NOT retry this search — proceed using only local knowledge or other available tools."
            )
        return f"[Web Search Error] HTTP {e.code}: {e.reason}. Do not retry — proceed with available information."
    except urllib.error.URLError as e:
        return f"[Web Search Unavailable] Network error contacting Serper: {e.reason}. Do not retry — proceed with available information."

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
        results.append(f"[{i + 1}] {title}{date_str}\n{snippet}\nURL: {link}\n")

    if len(results) <= 1:
        return (
            f"🔎 [Source: Serper Web Search]\n"
            f"No results found for '{final_query}'. "
            "This topic may not exist or may be too new/niche. "
            "Do NOT retry with the same or similar query — proceed with the information you already have."
        )

    return "\n".join(results)


def _perform_tavily_search(query: str, api_key: str, num_results: int = 5) -> str:
    if not api_key or api_key == "EMPTY":
        return ""

    try:
        payload = json.dumps({
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": num_results,
            "include_answer": True,
        }).encode("utf-8")

        req = urllib.request.Request("https://api.tavily.com/search", method="POST")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, data=payload, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        items = data.get("results", [])
        if not items:
            return ""

        blocks = ["🔎 [Source: Tavily Search (Serper fallback)]"]
        if data.get("answer"):
            blocks.append(f"⭐ [Direct Answer]: {data['answer']}\n")

        for i, item in enumerate(items):
            title = item.get("title", "")
            content = (item.get("content") or item.get("snippet") or "")[:250]
            url = item.get("url", "")
            blocks.append(f"[{i + 1}] {title}\n{content}\nURL: {url}\n")

        return "\n".join(blocks)
    except urllib.error.HTTPError as e:
        console.print(f"[dim]Tavily fallback HTTP {e.code}: check TAVILY_API_KEY[/dim]")
        return ""
    except Exception as e:
        console.print(f"[dim]Tavily fallback failed: {e}[/dim]")
        return ""


def wikimedia_search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    limit = max(1, min(int(top_k), 10))
    req = urllib.request.Request(
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={quote(query)}&srlimit={limit}&format=json&utf8=1",
        method="GET",
    )
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "BatchAgent/1.0")

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
            f"[YT-{i + 1}] {item.get('title', '').strip()}\n"
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
            f"[WM-{i + 1}] {item.get('title', '').strip()}\n"
            f"{item.get('description', '').strip()}\n"
            f"URL: {item.get('url', '').strip()}\n"
            f"Source: {item.get('source', 'Wikimedia/MediaWiki').strip()}\n"
        )
    return "\n".join(blocks)


def _is_serper_error(text: str) -> bool:
    return text.startswith("[Web Search Unavailable]") or text.startswith("[Web Search Error]")


def _perform_document_rag_search(
    query: str,
    top_k: int = 3,
    document_search_fn: Optional[Callable[[str, int], str]] = None,
) -> str:
    if document_search_fn is None:
        return ""
    try:
        return document_search_fn(query, top_k)
    except Exception as e:
        console.print(f"[dim]Document RAG search skipped: {e}[/dim]")
        return ""


# =============================================================================
# Main search entry
# =============================================================================

def perform_domain_aware_search(
    query: str,
    category: str,
    serper_api_key: str,
    current_time: str = "",
    enable_youtube: bool = False,
    tavily_api_key: str = "",
    document_search_fn: Optional[Callable[[str, int], str]] = None,
) -> str:
    """
    Aggregated search entry.

    Combines:
    - Serper domain-aware search
    - Tavily fallback
    - Wikimedia search
    - optional current-document RAG
    - optional YouTube search + extraction
    """
    domain_filters: Dict[str, str] = {
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
        "code":         "site:github.com OR site:stackoverflow.com OR site:docs.python.org OR site:pypi.org OR site:realpython.com",
        "finance":      "site:sec.gov OR site:finance.yahoo.com OR site:bloomberg.com OR site:investopedia.com",
        "health":       "site:pubmed.ncbi.nlm.nih.gov OR site:nih.gov OR site:mayoclinic.org OR site:webmd.com",
        "programming":  "site:stackoverflow.com OR site:github.com OR site:realpython.com OR site:docs.python.org",
        "research":     "site:arxiv.org OR site:semanticscholar.org OR site:scholar.google.com OR site:jstor.org",
        "general":      "",
    }

    aliases: Dict[str, str] = {
        "software": "software_eng",
        "software_engineering": "software_eng",
        "engineering": "software_eng",
        "medicine": "medical",
        "biology": "medical",
        "physics": "science",
        "chemistry": "science",
        "mathematics": "math",
        "maths": "math",
        "statistics": "math",
        "economics": "business",
        "stock": "business",
        "stocks": "business",
        "investment": "business",
        "support": "sales_support",
        "crm": "sales_support",
        "system": "assistant",
        "computer": "assistant",
        "paper": "academic",
        "papers": "academic",
        "python": "code",
        "javascript": "code",
        "js": "code",
    }

    cat = aliases.get(category.lower().strip(), category.lower().strip())
    if cat not in domain_filters:
        cat = "general"

    filter_str = domain_filters[cat]

    reference_year = _extract_reference_year(current_time)
    normalized_query = query.strip()
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", normalized_query))
    has_relative_time = bool(re.search(r"\b(latest|newest|current|today|this year|recent)\b", normalized_query, re.I))

    if reference_year and not has_year and has_relative_time:
        normalized_query = f"{normalized_query} in {reference_year}"
    elif reference_year and not has_year:
        normalized_query = f"{normalized_query} as of {reference_year}"

    final_query = f"{normalized_query} {filter_str}".strip() if filter_str else normalized_query

    serper_enabled = bool(serper_api_key) and serper_api_key != "EMPTY"
    yt_enabled = enable_youtube and serper_enabled

    console.print(
        f"[dim]Search -> cat='{cat}' serper={'✓' if serper_enabled else '✗'} "
        f"youtube={'✓' if yt_enabled else '✗'} tavily={'✓' if tavily_api_key else '✗'}[/dim]"
    )

    web_output = ""
    wikimedia_items: List[Dict[str, str]] = []
    doc_output = ""
    youtube_items: List[Dict[str, str]] = []

    try:
        futures: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            if serper_enabled:
                futures["web"] = executor.submit(_perform_serper_domain_search, final_query, cat, serper_api_key)

            futures["wm"] = executor.submit(wikimedia_search, normalized_query, _resolve_youtube_top_k(None))
            futures["doc"] = executor.submit(_perform_document_rag_search, normalized_query, 3, document_search_fn)

            if yt_enabled:
                futures["yt"] = executor.submit(youtube_search_and_extract, normalized_query, serper_api_key, None)

            web_output = futures["web"].result() if "web" in futures else ""
            wikimedia_items = futures["wm"].result()
            doc_output = futures["doc"].result()
            youtube_items = futures["yt"].result() if "yt" in futures else []
    except Exception as e:
        console.print(f"[red]Search parallel fetch error: {e}[/red]")

    if _is_serper_error(web_output) or (not serper_enabled):
        if tavily_api_key:
            console.print("[dim]Serper unavailable — trying Tavily fallback...[/dim]")
            tavily_result = _perform_tavily_search(normalized_query, tavily_api_key)
            if tavily_result:
                web_output = tavily_result

        if not web_output:
            web_output = (
                "[Web Search Unavailable] Neither Serper nor Tavily is configured. "
                "Set SERPER_API_KEY or TAVILY_API_KEY to enable web search."
            )

    outputs: List[str] = []
    if web_output:
        outputs.append(web_output)
    if doc_output:
        outputs.append(doc_output)
    if yt_enabled:
        yt_output = _format_youtube_search_results(youtube_items)
        if yt_output:
            outputs.append(yt_output)

    wm_output = _format_wikimedia_search_results(wikimedia_items)
    if wm_output:
        outputs.append(wm_output)

    return "\n\n".join(outputs) if outputs else "No search results available."