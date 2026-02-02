#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import kenlm


# =============================
# Config
# =============================
@dataclass
class FixConfig:
    avg_token_len_threshold: float = 7.0
    long_token_len_threshold: int = 18
    long_token_ratio_threshold: float = 0.18

    min_glued_token_len: int = 16

    beam_size: int = 32
    max_word_len: int = 24
    allow_unknown_words: bool = True
    unknown_word_penalty: float = 2.0
    word_count_penalty: float = 0.15

    accept_delta_avg: float = 0.0025  # slightly less strict than v1

    # Do NOT split letter+digit (Qwen3). We'll keep it intact by default.
    split_letter_digit: bool = False

    protect_urls_emails: bool = True
    protect_citations: bool = True
    protect_model_like_tokens: bool = True


# =============================
# Vocabulary (extend as needed)
# =============================
COMMON_WORDS = set("""
a an the and or of to in on for with from by as at into
this that these those it its we our you your they their
work introduce introduces introduced present presents
latest version model models family series include includes comprised comprises
large language multilingual capabilities architecture architectures
dense mixture expert experts moe reasoning thinking mode modes
rapid context driven response responses unified framework
eliminates eliminate need switch between different such chat optimized
dedicated enables dynamic switching based user queries templates
meanwhile budget mechanism allowing allow allocate computational resources
adaptively during inference thereby balancing latency performance based
task complexity moreover leveraging knowledge flagship significantly reduce
required build smaller scale while ensuring highly competitive empirical
evaluations demonstrate achieves state art results across diverse benchmarks
including tasks code generation mathematical agent competitive against larger
proprietary compared predecessor expands multilingual support languages dialects
enhancing global accessibility improved cross lingual understanding generation
facilitate reproducibility community research development publicly accessible
under apache
""".split())


def dict_dp_with_unknown_prefix(token: str, cfg: FixConfig) -> Optional[List[str]]:
    """
    Allow an unknown prefix (e.g., 'Qwen') followed by fully coverable common words.
    Good for: Qwenmodelfamily -> Qwen model family
    """
    if len(token) < cfg.min_glued_token_len or not token.isalpha():
        return None

    # Heuristic: unknown prefix is a leading Capitalized chunk
    m = re.match(r"^([A-Z][a-zA-Z]{2,12})([a-z].+)$", token)
    if not m:
        return None

    prefix = m.group(1)
    rest = m.group(2)

    # Run your existing dict DP on rest (lowercase)
    lower = rest.lower()
    n = len(lower)
    dp: List[Optional[List[str]]] = [None] * (n + 1)
    dp[0] = []
    for i in range(n):
        if dp[i] is None:
            continue
        for j in range(i + 1, min(n, i + cfg.max_word_len) + 1):
            piece = lower[i:j]
            if piece in COMMON_WORDS:
                cand = dp[i] + [piece]
                if dp[j] is None or len(cand) < len(dp[j]):
                    dp[j] = cand

    if dp[n] is None or len(dp[n]) <= 1:
        return None

    return [prefix] + dp[n]

# =============================
# Regex helpers
# =============================
PUNCT = r"""[,.;:!?()\[\]{}]"""

MODEL_TOKEN_RE = re.compile(
    # Examples: Qwen3, Qwen2.5, GPT-4o, QwQ-32B, DeepSeek-R1, Qwen3-235B-A22B
    r"\b("
    r"[A-Za-z]{1,12}\d+(?:\.\d+)?(?:[-_][A-Za-z0-9]{1,16})*"
    r"|"
    r"[A-Za-z]{2,12}(?:[-_]\d+[A-Za-z0-9]{0,8})+"
    r")\b"
)

CITATION_RE = re.compile(r"\([A-Za-z][A-Za-z .-]*,\s*\d{4}[a-z]?\)")
URL_RE = re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# short CamelCase like MoE / QwQ should be protected
SHORT_CAMEL_RE = re.compile(r"\b[A-Z][a-z]?[A-Z][A-Za-z]?\b")  # MoE, QwQ, etc.


def normalize_unicode(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return text


def normalize_hyphenation(text: str) -> str:
    return re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def mild_punct_cleanup(text: str) -> str:
    """
    Conservative punctuation cleanup for PDF text:
    - Remove spaces before .,;:!? but keep after
    - Normalize spaces around parentheses/brackets
    - Fix "from ." -> "from."
    - Fix stray "]" spacing: "its ] predecessor" -> "its] predecessor" then later add proper spacing if needed
    """
    t = text

    # collapse spaces before punctuation
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)

    # tighten opening brackets and loosen closing appropriately
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\[\s+", "[", t)
    t = re.sub(r"\s+\]", "]", t)

    # common artifact: "from ." -> "from."
    t = re.sub(r"\bfrom\s+\.", "from.", t)

    # normalize multi-spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def conservative_boundary_spacing(text: str, cfg: FixConfig) -> str:
    """
    Only apply safe boundary spacing:
    - lower->Upper boundaries (wepresentQwen -> wepresent Qwen)
    - optional letter<->digit boundaries (disabled by default, because Qwen3 should remain intact)
    - punctuation followed by letter: "etc.,competitive" -> "etc., competitive"
    """
    t = text

    # lower->Upper (safe)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)

    if cfg.split_letter_digit:
        t = re.sub(r"([A-Za-z]{3,})(\d+)", r"\1 \2", t)
        t = re.sub(r"(\d+)([A-Za-z]{3,})", r"\1 \2", t)

    t = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", t)

    return t


# =============================
# Protect spans from being changed
# =============================
def protect_spans(text: str, cfg: FixConfig) -> Tuple[str, Dict[str, str]]:
    replacements: Dict[str, str] = {}
    idx = 0

    def replace_match(m, tag):
        nonlocal idx
        key = f"@@{tag}_{idx}@@"
        idx += 1
        replacements[key] = m.group(0)
        return key

    if cfg.protect_urls_emails:
        text = URL_RE.sub(lambda m: replace_match(m, "URL"), text)
        text = EMAIL_RE.sub(lambda m: replace_match(m, "EMAIL"), text)

    if cfg.protect_citations:
        text = CITATION_RE.sub(lambda m: replace_match(m, "CITE"), text)

    if cfg.protect_model_like_tokens:
        text = MODEL_TOKEN_RE.sub(lambda m: replace_match(m, "MODEL"), text)
        text = SHORT_CAMEL_RE.sub(lambda m: replace_match(m, "CAMEL"), text)

    return text, replacements


def restore_spans(text: str, replacements: Dict[str, str]) -> str:
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


# =============================
# Suspicious paragraph detection
# =============================
def is_suspicious_paragraph(text: str, cfg: FixConfig) -> bool:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return False
    lengths = [len(t) for t in tokens]
    avg_len = sum(lengths) / len(lengths)
    long_tokens = [l for l in lengths if l >= cfg.long_token_len_threshold]
    ratio = len(long_tokens) / max(len(tokens), 1)
    return (avg_len >= cfg.avg_token_len_threshold) or (ratio >= cfg.long_token_ratio_threshold)


# =============================
# Tokenization
# Keep protected placeholders intact.
# =============================
# TOKEN_RE = re.compile(
#     rf"@@[A-Z]+_\d+@@|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|{PUNCT}|--+|[-]+|[^\s]+"
# )

TOKEN_RE = re.compile(
    rf"@@[A-Z]+_\d+@@"
    rf"|[A-Za-z]+(?:\d+(?:\.\d+)?)?(?:[-_][A-Za-z0-9]+)*"   # keep Qwen3, Qwen2.5, GPT-4o, QwQ-32B, Qwen3-235B-A22B
    rf"|\d+(?:\.\d+)?"
    rf"|{PUNCT}"
    rf"|--+"
    rf"|[-]+"
    rf"|[^\s]+"
)

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def detokenize(tokens: List[str]) -> str:
    s = " ".join(tokens)
    s = mild_punct_cleanup(s)
    return s


# =============================
# KenLM scoring
# =============================
def kenlm_avg_score(model: kenlm.Model, text: str) -> Tuple[float, float, int]:
    fs = list(model.full_scores(text, bos=True, eos=True))
    n = max(len(fs), 1)
    total = model.score(text, bos=True, eos=True)
    return total / n, total, n


# =============================
# Dictionary DP segmentation
# =============================
def dict_dp_segment(token: str, cfg: FixConfig) -> Optional[List[str]]:
    if len(token) < cfg.min_glued_token_len:
        return None
    if not token.isalpha():
        return None

    original_is_cap = token[0].isupper()
    lower = token.lower()
    n = len(lower)

    dp: List[Optional[List[str]]] = [None] * (n + 1)
    dp[0] = []
    for i in range(n):
        if dp[i] is None:
            continue
        for j in range(i + 1, min(n, i + cfg.max_word_len) + 1):
            piece = lower[i:j]
            if piece in COMMON_WORDS:
                cand = dp[i] + [piece]
                if dp[j] is None or len(cand) < len(dp[j]):
                    dp[j] = cand

    if dp[n] is None or len(dp[n]) <= 1:
        return None

    words = dp[n]
    if original_is_cap:
        words[0] = words[0].capitalize()
    return words


# =============================
# KenLM beam-search segmentation (single alphabetic token)
# =============================
def kenlm_beam_segment(model: kenlm.Model, token: str, cfg: FixConfig) -> Optional[List[str]]:
    if len(token) < cfg.min_glued_token_len:
        return None
    if not token.isalpha():
        return None

    original_is_cap = token[0].isupper()
    t = token.lower()
    n = len(t)

    BeamState = Tuple[int, List[str], float]  # pos, words, cost
    beam: List[BeamState] = [(0, [], 0.0)]

    def piece_penalty(piece: str) -> float:
        if piece in COMMON_WORDS:
            return 0.0
        if not cfg.allow_unknown_words:
            return 1e9
        return cfg.unknown_word_penalty

    for _ in range(n + 1):
        cand_states: List[BeamState] = []
        for pos, words, _cost in beam:
            if pos >= n:
                cand_states.append((pos, words, _cost))
                continue

            for j in range(pos + 1, min(n, pos + cfg.max_word_len) + 1):
                piece = t[pos:j]

                # prune tiny fragments
                if len(piece) == 1 and piece not in COMMON_WORDS:
                    continue
                if len(piece) == 2 and piece not in COMMON_WORDS and j < n:
                    continue

                pen = piece_penalty(piece)
                if pen > 1e8:
                    continue

                new_words = words + [piece]
                phrase = " ".join(new_words)

                avg, _, _ = kenlm_avg_score(model, phrase)
                # higher avg is better => cost is negative avg
                lm_cost = -avg
                wc_cost = cfg.word_count_penalty * len(new_words)

                cost = lm_cost + wc_cost + pen
                cand_states.append((j, new_words, cost))

        cand_states.sort(key=lambda x: x[2])
        beam = cand_states[: cfg.beam_size]

        if all(pos >= n for pos, _, _ in beam):
            break

    complete = [(words, cost) for pos, words, cost in beam if pos >= n]
    if not complete:
        return None
    complete.sort(key=lambda x: x[1])
    best_words = complete[0][0]
    if len(best_words) <= 1:
        return None

    if original_is_cap:
        best_words[0] = best_words[0].capitalize()
    return best_words


def choose_best_rewrite_for_token(model: kenlm.Model, token: str, cfg: FixConfig) -> Optional[str]:
    # only handle alphabetic glued tokens
    if not token.isalpha() or len(token) < cfg.min_glued_token_len:
        return None

    base_avg, _, _ = kenlm_avg_score(model, token)

    candidates: List[Tuple[str, float]] = []

    dp_words = dict_dp_segment(token, cfg)
    if dp_words:
        cand = " ".join(dp_words)
        avg, _, _ = kenlm_avg_score(model, cand)
        candidates.append((cand, avg))

    beam_words = kenlm_beam_segment(model, token, cfg)
    if beam_words:
        cand = " ".join(beam_words)
        avg, _, _ = kenlm_avg_score(model, cand)
        candidates.append((cand, avg))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_text, best_avg = candidates[0]

    if (best_avg - base_avg) >= cfg.accept_delta_avg:
        return best_text
    return None


# =============================
# Main fix routine
# =============================
def fix_text(model: kenlm.Model, text: str, cfg: FixConfig) -> str:
    t = normalize_unicode(text)
    t = normalize_hyphenation(t)
    t = normalize_whitespace(t)

    # protect model tokens, citations, etc. BEFORE inserting spaces
    t, repl = protect_spans(t, cfg)

    # safe boundary spacing and mild punct cleanup
    t = conservative_boundary_spacing(t, cfg)
    t = mild_punct_cleanup(t)

    # paragraph pass
    paragraphs = re.split(r"\n{2,}", t)
    out_paras: List[str] = []

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue

        if not is_suspicious_paragraph(p, cfg):
            out_paras.append(p)
            continue

        toks = tokenize(p)
        new_toks: List[str] = []

        for tok in toks:
            # keep protected placeholders intact
            if tok.startswith("@@") and tok.endswith("@@"):
                new_toks.append(tok)
                continue

            # only attempt segmentation for long alphabetic tokens
            if tok.isalpha() and len(tok) >= cfg.min_glued_token_len:
                rewritten = choose_best_rewrite_for_token(model, tok, cfg)
                if rewritten:
                    new_toks.extend(rewritten.split())
                else:
                    new_toks.append(tok)
            else:
                new_toks.append(tok)

        out_paras.append(detokenize(new_toks))

    out = "\n\n".join(out_paras)

    # restore protected spans
    out = restore_spans(out, repl)

    # final cleanup: fix "] predecessor" spacing to "predecessor"
    out = re.sub(r"\s*\]\s*", "] ", out)
    out = re.sub(r"\s+", " ", out).strip()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to KenLM .bin/.arpa.bin")
    ap.add_argument("--text", required=True, help="Input text")
    ap.add_argument("--delta", type=float, default=0.0025, help="Accept delta avg threshold")
    ap.add_argument("--beam", type=int, default=32, help="Beam size")
    args = ap.parse_args()

    cfg = FixConfig(accept_delta_avg=args.delta, beam_size=args.beam)
    m = kenlm.Model(args.model)

    fixed = fix_text(m, args.text, cfg)

    print("\n================ ORIGINAL ================\n")
    print(args.text)
    print("\n================= FIXED ==================\n")
    print(fixed)
    print("\n=========================================\n")


if __name__ == "__main__":
    main()