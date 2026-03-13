import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None  # type: ignore[assignment]

try:
    import kenlm  # type: ignore[import]
except Exception:
    kenlm = None  # type: ignore[assignment]

try:
    from scripts import fix_glued_text_with_kenlm_v3 as gluefix  # type: ignore[import]
except Exception:
    gluefix = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import]
except Exception:
    AutoModelForSeq2SeqLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


COMMON_WORDS = {
    "a",
    "about",
    "across",
    "advanced",
    "after",
    "against",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "align",
    "application",
    "applications",
    "based",
    "better",
    "been",
    "before",
    "between",
    "both",
    "but",
    "by",
    "behavior",
    "behaviour",
    "capabilities",
    "can",
    "case",
    "certain",
    "collection",
    "complex",
    "comprises",
    "computational",
    "context",
    "data",
    "dataset",
    "datasets",
    "designed",
    "development",
    "different",
    "domain",
    "domains",
    "downstream",
    "efficiency",
    "effective",
    "effective",
    "effort",
    "efficient",
    "efficiently",
    "enhance",
    "evaluation",
    "evaluations",
    "for",
    "from",
    "function",
    "further",
    "global",
    "goal",
    "foundation",
    "have",
    "how",
    "human",
    "in",
    "including",
    "intelligence",
    "introduction",
    "into",
    "is",
    "its",
    "key",
    "knowledge",
    "language",
    "languages",
    "large",
    "latest",
    "learning",
    "local",
    "long",
    "many",
    "model",
    "models",
    "mode",
    "multilingual",
    "new",
    "not",
    "of",
    "on",
    "one",
    "open",
    "our",
    "parameters",
    "performance",
    "preferences",
    "present",
    "progress",
    "provide",
    "provides",
    "rapid",
    "reasoning",
    "recent",
    "reduce",
    "reported",
    "research",
    "resource",
    "resources",
    "results",
    "specific",
    "scaling",
    "series",
    "smaller",
    "state",
    "support",
    "suit",
    "suits",
    "tasks",
    "than",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "through",
    "time",
    "to",
    "token",
    "tokens",
    "toward",
    "training",
    "understanding",
    "thinking",
    "meet",
    "varying",
    "complexity",
    "real",
    "world",
    "use",
    "used",
    "user",
    "users",
    "various",
    "we",
    "while",
    "with",
    "work",
}


@dataclass
class TextCleanerConfig:
    avg_token_len_threshold: float = 8.5
    long_token_len_threshold: int = 16
    long_token_ratio_threshold: float = 0.2
    min_glued_token_len: int = 8
    use_kenlm: bool = True
    use_seq2seq: bool = False
    seq2seq_model_name: str = "t5-small"


_KENLM_MODEL = None
_SEQ2SEQ_MODEL = None
_SEQ2SEQ_TOKENIZER = None


def _debug_enabled() -> bool:
    val = os.getenv("APP_DEBUG", "")
    return val.lower() in ("1", "true", "yes", "on")


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[text_pipeline] {msg}")


def _normalize_paragraph(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_suspicious(text: str, cfg: TextCleanerConfig) -> bool:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return False
    lengths: List[int] = [len(t) for t in tokens]
    avg_len = sum(lengths) / len(lengths)
    long_tokens = [l for l in lengths if l >= cfg.long_token_len_threshold]
    ratio = len(long_tokens) / len(lengths)
    return avg_len >= cfg.avg_token_len_threshold or ratio >= cfg.long_token_ratio_threshold


def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")
    text = text.replace("\u2007", " ")
    text = text.replace("\u202F", " ")
    for dash in ["\u2012", "\u2013", "\u2014", "\u2015", "\u2212"]:
        text = text.replace(dash, "-")
    return text


def _fix_line_hyphens(text: str) -> str:
    pattern = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
    return pattern.sub("", text)


def _fix_urls(text: str) -> str:
    text = re.sub(r"\b(https?)\s*:\s*/\s*/\s*", lambda m: f"{m.group(1).lower()}://", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(\bhttps?://[A-Za-z0-9_.-]+)\s*\.\s*([A-Za-z]{2,})(\s*/)",
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


def _normalize_punct_spacing(text: str) -> str:
    text = re.sub(r"\s+([,;:!?()\[\]{}])", r"\1", text)
    text = re.sub(r"([,;:!?()\[\]{}])(?=\S)", r"\1 ", text)
    text = re.sub(r"(?<!\d)\.(?=[A-Za-z])", r". ", text)
    text = re.sub(r"\s+([,;:!?()\[\]{}])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bo3\(", "o3 (", text)
    return text.strip()


def _word_cost(piece: str) -> float:
    if zipf_frequency is None:
        return 1.0
    freq = zipf_frequency(piece, "en")  # type: ignore[arg-type]
    if freq <= 0.0:
        return 8.0
    return 7.0 - freq


def _load_kenlm_model():
    global _KENLM_MODEL
    if _KENLM_MODEL is not None:
        return _KENLM_MODEL
    if kenlm is None:
        return None
    import os

    path = os.getenv("KENLM_MODEL_PATH")
    if not path:
        return None
    try:
        _KENLM_MODEL = kenlm.Model(path)  # type: ignore[call-arg]
    except Exception:
        _KENLM_MODEL = None
    return _KENLM_MODEL


def _kenlm_score(text: str) -> float:
    model = _load_kenlm_model()
    if model is None:
        return 0.0
    return float(model.score(text))


def _kenlm_glue_fix_paragraph(text: str, cfg: TextCleanerConfig) -> str:
    if gluefix is None:
        return text
    model = _load_kenlm_model()
    if model is None:
        return text
    fix_cfg = gluefix.FixConfig(debug=_debug_enabled())
    p = text.strip()
    if not p:
        return text
    return gluefix.fix_text(model, p, fix_cfg)


def _load_seq2seq_model(model_name: str):
    global _SEQ2SEQ_MODEL, _SEQ2SEQ_TOKENIZER
    if _SEQ2SEQ_MODEL is not None and _SEQ2SEQ_TOKENIZER is not None:
        return _SEQ2SEQ_MODEL, _SEQ2SEQ_TOKENIZER
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        return None, None
    env_name = os.getenv("APP_TEXT_SEQ2SEQ_MODEL")
    if env_name:
        model_name = env_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        return None, None
    _SEQ2SEQ_MODEL = model
    _SEQ2SEQ_TOKENIZER = tokenizer
    return model, tokenizer


def _seq2seq_correct(text: str, cfg: TextCleanerConfig) -> str:
    model, tokenizer = _load_seq2seq_model(cfg.seq2seq_model_name)
    if model is None or tokenizer is None:
        return text
    prefix = os.getenv("APP_TEXT_SEQ2SEQ_PREFIX")
    if prefix:
        inp = prefix + " " + text
    else:
        inp = text
    inputs = tokenizer([inp], return_tensors="pt", truncation=True)
    try:
        outputs = model.generate(**inputs, max_new_tokens=inputs["input_ids"].shape[1] + 32)
    except Exception:
        return text
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if not decoded:
        return text
    result = decoded[0]
    if _debug_enabled() and result != text:
        _debug(f"seq2seq: {text[:120].replace(chr(10), ' ')} -> {result[:120].replace(chr(10), ' ')}")
    return result


def _segment_token_with_kenlm(token: str) -> str:
    model = _load_kenlm_model()
    if model is None:
        return token
    lower = token.lower()
    n = len(lower)
    best_split = None
    best_score = None
    for i in range(2, n - 1):
        left = lower[:i]
        right = lower[i:]
        if not left.isalpha() or not right.isalpha():
            continue
        cand = f"{left} {right}"
        score = _kenlm_score(cand)
        if best_score is None or score > best_score:
            best_score = score
            best_split = [left, right]
    for i in range(2, n - 2):
        for j in range(i + 1, n - 1):
            p1 = lower[:i]
            p2 = lower[i:j]
            p3 = lower[j:]
            if not p1.isalpha() or not p2.isalpha() or not p3.isalpha():
                continue
            cand = f"{p1} {p2} {p3}"
            score = _kenlm_score(cand)
            if best_score is None or score > best_score:
                best_score = score
                best_split = [p1, p2, p3]
    if not best_split or len(best_split) <= 1:
        return token
    out = []
    for idx, w in enumerate(best_split):
        if idx == 0:
            out.append(w.capitalize() if token[0].isupper() else w)
        else:
            out.append(w)
    return " ".join(out)


def _segment_mixed_token(token: str, cfg: TextCleanerConfig) -> str:
    idx = token.find("Qwen")
    if idx > 0:
        j = idx + 4
        n = len(token)
        while j < n and (token[j].isalnum() or token[j] == "-"):
            j += 1
        prefix = token[:idx]
        model = token[idx:j]
        suffix = token[j:]
        parts: List[str] = []
        if prefix:
            if prefix.isalpha() and len(prefix) >= cfg.min_glued_token_len:
                parts.append(_segment_alpha_token(prefix, cfg))
            else:
                parts.append(prefix)
        parts.append(model)
        if suffix:
            if suffix.isalpha() and len(suffix) >= cfg.min_glued_token_len:
                parts.append(_segment_alpha_token(suffix, cfg))
            else:
                parts.append(suffix)
        return " ".join(p for p in parts if p)
    pieces: List[str] = []
    i = 0
    n = len(token)
    while i < n:
        ch = token[i]
        if ch.isalpha():
            j = i
            while j < n and token[j].isalpha():
                j += 1
            core = token[i:j]
            if len(core) >= cfg.min_glued_token_len:
                pieces.append(_segment_alpha_token(core, cfg))
            else:
                pieces.append(core)
            i = j
        else:
            pieces.append(ch)
            i += 1
    return "".join(pieces)


def _segment_alpha_token(token: str, cfg: TextCleanerConfig) -> str:
    if len(token) < cfg.min_glued_token_len:
        return token
    lower = token.lower()
    n = len(lower)
    dp_cost: List[float] = [float("inf")] * (n + 1)
    dp_seg: List[Optional[List[str]]] = [None] * (n + 1)
    dp_cost[0] = 0.0
    dp_seg[0] = []
    for i in range(n):
        seg_i = dp_seg[i]
        if seg_i is None:
            continue
        for j in range(i + 2, n + 1):
            piece = lower[i:j]
            if not piece.isalpha():
                continue
            if piece not in COMMON_WORDS and zipf_frequency is None:
                continue
            cost = _word_cost(piece)
            new_cost = dp_cost[i] + cost
            if new_cost < dp_cost[j]:
                dp_cost[j] = new_cost
                dp_seg[j] = seg_i + [piece]
    if dp_seg[n] is None or len(dp_seg[n]) <= 1:
        if token[0].lower() == "s" and len(token) > cfg.min_glued_token_len + 1:
            inner = _segment_alpha_token(token[1:], cfg)
            if inner != token[1:]:
                return token[0] + " " + inner
        if cfg.use_kenlm:
            return _segment_token_with_kenlm(token)
        return token
    words = dp_seg[n] or []
    out = []
    for idx, w in enumerate(words):
        if idx == 0:
            out.append(w.capitalize() if token[0].isupper() else w)
        else:
            out.append(w)
    result = " ".join(out)
    if _debug_enabled() and result != token:
        _debug(f"segment_token: {token} -> {result}")
    return result


def _segment_token(token: str, cfg: TextCleanerConfig) -> str:
    if len(token) < cfg.min_glued_token_len:
        return token
    if token.isalpha():
        return _segment_alpha_token(token, cfg)
    return _segment_mixed_token(token, cfg)


def _merge_broken_alpha_tokens(tokens: List[str], cfg: TextCleanerConfig) -> List[str]:
    merged: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.isalpha() and len(tok) >= cfg.min_glued_token_len:
            combined = tok
            j = i + 1
            while j < n:
                nxt = tokens[j]
                if not nxt.isalpha():
                    break
                if len(nxt) > 4:
                    break
                if nxt.lower() in COMMON_WORDS:
                    break
                combined += nxt
                j += 1
            if j > i + 1:
                merged.append(combined)
                i = j
                continue
        merged.append(tok)
        i += 1
    return merged


def _clean_paragraph(text: str, cfg: TextCleanerConfig) -> str:
    base = _normalize_paragraph(text)
    if not base:
        return ""
    marked = re.sub(r"([,.;:!?()\-\[\]])", r" \1 ", base)
    tokens = marked.split()
    filtered_tokens: List[str] = []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha() and tok.isupper() and tok not in ("I", "A"):
            continue
        filtered_tokens.append(tok)
    tokens = _merge_broken_alpha_tokens(filtered_tokens, cfg)
    model_available = _load_kenlm_model() is not None
    if model_available:
        cleaned = " ".join(tokens)
    else:
        cleaned_tokens: List[str] = []
        for tok in tokens:
            if len(tok) >= cfg.min_glued_token_len:
                cleaned_tokens.append(_segment_token(tok, cfg))
            else:
                cleaned_tokens.append(tok)
        cleaned = " ".join(cleaned_tokens)
    normalized = _normalize_punct_spacing(cleaned)
    if cfg.use_kenlm:
        normalized = _kenlm_glue_fix_paragraph(normalized, cfg)
        tokens2: List[str] = []
        for tok in normalized.split():
            m = re.match(r"^([^A-Za-z]*)([A-Za-z]+)(.*)$", tok)
            if m and len(m.group(2)) >= cfg.long_token_len_threshold:
                core = _segment_token(m.group(2), cfg)
                tokens2.append(m.group(1) + core + m.group(3))
            else:
                tokens2.append(tok)
        normalized = " ".join(tokens2)
        threshold = max(int(cfg.long_token_len_threshold * 1.3), 22)
        long_core_re = re.compile(rf"[A-Za-z]{{{threshold},}}")
        def _repl_long(match: re.Match[str]) -> str:
            token = match.group(0)
            return _segment_token(token, cfg)
        normalized = long_core_re.sub(_repl_long, normalized)
    if cfg.use_seq2seq and _is_suspicious(base, cfg):
        normalized = _seq2seq_correct(normalized, cfg)
    if _debug_enabled():
        _debug(f"paragraph in: {base[:200]}")
        _debug(f"paragraph out: {normalized[:200]}")
    return normalized


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def clean_document_text(text: str, cfg: TextCleanerConfig | None = None) -> str:
    if cfg is None:
        cfg = TextCleanerConfig(
            use_kenlm=_env_bool("APP_TEXT_USE_KENLM", True),
            use_seq2seq=_env_bool("APP_TEXT_USE_SEQ2SEQ", False),
        )
    text = _normalize_unicode(text)
    text = _fix_line_hyphens(text)
    text = _fix_urls(text)
    text = text.replace("\r", "\n")
    paragraphs = re.split(r"\n{2,}", text)
    cleaned_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        cleaned_paragraphs.append(_clean_paragraph(para, cfg))
    return "\n\n".join(cleaned_paragraphs)
