#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import kenlm

try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None  # type: ignore[assignment]


# =============================
# Config
# =============================
@dataclass
class FixConfig:
    # suspicious paragraph detection
    avg_token_len_threshold: float = 6.5
    long_token_len_threshold: int = 16
    long_token_ratio_threshold: float = 0.16

    # default trigger for "long" glued tokens
    min_glued_token_len: int = 10

    # short-glue trigger (for tokens like "wepresent")
    short_glue_min_len: int = 5
    short_glue_max_len: int = 18
    short_glue_prefixes: Tuple[str, ...] = (
        "we",
        "in",
        "from",
        "this",
        "that",
        "there",
        "into",
        "with",
        "while",
        "most",
        "their",
        "these",
        "those",
        "ther",
        "the",
        "and",
        "step",
        "thinking",
        "for",
        "on",
        "our",
        "moreover",
        "additionally",
        "expands",
        "expand",
    )

    # beam search
    beam_size: int = 48
    max_word_len: int = 24
    allow_unknown_words: bool = True
    unknown_word_penalty: float = 1.0     # allow unknown prefixes like Qwen
    word_count_penalty: float = 0.15      # discourage over-splitting

    # accept segmentation only if it improves LM avg score by >= threshold
    accept_delta_avg: float = 0.0015

    # Do NOT split letter+digit (Qwen3) at rule stage
    split_letter_digit: bool = False

    # protection switches
    protect_urls_emails: bool = True
    protect_citations: bool = True
    protect_model_like_tokens: bool = True

    # debug
    debug: bool = False


# =============================
# Vocabulary (extend later via corpus-derived list)
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
real world application applications
multi step non thinking mode
both
while most remain rapid growth open source communities substantially reduced performance gap between weight
flexibility ensures ensure developers developer users user adapt behavior specific tasks task efficiently efficiency
also generate generates synthetic data using domain domains specific models model align foundation foundations human preference preferences downstream application applications employ employs multi stage post training
general domain reinforcement learning improve performance across wider range downstream tasks
incorporate incorporates incorporated incorporating budget budgets can agent agents related task tasks wide wider range align aligns aligned
""".split())


# =============================
# Regex helpers
# =============================
PUNCT = r"""[,.;:!?()\[\]{}]"""

# Model-ish tokens to protect: Qwen3, Qwen2.5, GPT-4o, QwQ-32B, DeepSeek-R1, Qwen3-235B-A22B
MODEL_TOKEN_RE = re.compile(
    r"\b("
    r"[A-Za-z]{1,12}\d+(?:\.\d+)?(?:[-_][A-Za-z0-9]{1,16})*"
    r"|"
    r"[A-Za-z]{2,12}(?:[-_]\d+[A-Za-z0-9]{0,8})+"
    r")\b"
)

CITATION_RE = re.compile(r"\([A-Za-z][A-Za-z .-]*,\s*\d{4}[a-z]?\)")
URL_RE = re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# short CamelCase like MoE / QwQ should be protected too (even without digits/hyphen)
SHORT_CAMEL_RE = re.compile(r"\b[A-Z][a-z]?[A-Z][A-Za-z]?\b")


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
    Conservative punctuation cleanup for PDF text.
    """
    t = text
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\[\s+", "[", t)
    t = re.sub(r"\s+\]", "]", t)
    t = re.sub(r"\bfrom\s+\.", "from.", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def conservative_boundary_spacing(text: str, cfg: FixConfig) -> str:
    """
    Only safe boundary spacing:
    - lower->Upper boundaries
    - optional letter<->digit boundaries (disabled by default)
    - punctuation followed by letter
    """
    t = text
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)

    if cfg.split_letter_digit:
        t = re.sub(r"([A-Za-z]{3,})(\d+)", r"\1 \2", t)
        t = re.sub(r"(\d+)([A-Za-z]{3,})", r"\1 \2", t)

    t = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", t)
    return t


# =============================
# Protect spans
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
# Key: prefer alnum/model-ish tokens BEFORE splitting letters and digits
# =============================
TOKEN_RE = re.compile(
    rf"@@[A-Z]+_\d+@@"
    rf"|[A-Za-z]+(?:\d+(?:\.\d+)?)?(?:[-_][A-Za-z0-9]+)*"   # Qwen3, Qwen2.5, GPT-4o, QwQ-32B, Qwen3-235B-A22B
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


def _hyphen_letter_positions(text: str) -> List[int]:
    positions = []
    count = 0
    for ch in text:
        if ch == "-":
            positions.append(count)
        elif ch.isalpha():
            count += 1
    return positions


def _reinject_hyphen(original: str, base: str, rewritten: str) -> str:
    words = rewritten.split()
    if len(words) <= 1:
        return rewritten
    positions = _hyphen_letter_positions(original)
    if not positions:
        return rewritten
    cum = []
    total = 0
    for w in words:
        total += len(w)
        cum.append(total)
    join_idx = set()
    for pos in positions:
        for idx, bound in enumerate(cum):
            if bound >= pos:
                if idx < len(words) - 1:
                    join_idx.add(idx)
                break
    out_words = []
    i = 0
    n = len(words)
    while i < n:
        if i in join_idx and i + 1 < n:
            out_words.append(words[i] + "-" + words[i + 1])
            i += 2
        else:
            out_words.append(words[i])
            i += 1
    return " ".join(out_words)


# =============================
# Short glue trigger
# =============================
def should_try_short_glue(token: str, cfg: FixConfig) -> bool:
    if not token.isalpha():
        return False
    lower = token.lower()
    if cfg.short_glue_min_len <= len(token) <= cfg.short_glue_max_len and lower.startswith(cfg.short_glue_prefixes):
        return True
    return False


# =============================
# Dictionary DP segmentation
# =============================
def dict_dp_segment(token: str, cfg: FixConfig) -> Optional[List[str]]:
    if not token.isalpha():
        return None
    if len(token) < min(cfg.min_glued_token_len, cfg.short_glue_min_len):
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


def dict_dp_with_unknown_prefix(token: str, cfg: FixConfig) -> Optional[List[str]]:
    if not token.isalpha():
        return None

    prefix: Optional[str]
    rest: Optional[str]

    if token.startswith("Qwen") and len(token) > 4 and token[4:].islower():
        prefix = "Qwen"
        rest = token[4:]
    else:
        m = re.match(r"^([A-Z][a-z]{2,12}?)([a-z].+)$", token)
        if not m:
            return None
        prefix = m.group(1)
        rest = m.group(2)

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
# KenLM beam-search segmentation (single alphabetic token)
# Supports unknown pieces with penalty (good for Qwen + model + family)
# =============================
def kenlm_beam_segment(model: kenlm.Model, token: str, cfg: FixConfig) -> Optional[List[str]]:
    if not token.isalpha():
        return None

    # Only run beam if token is long glue or short-glue trigger
    if not (len(token) >= cfg.min_glued_token_len or should_try_short_glue(token, cfg)):
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

                # prune tiny fragments unless common
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


# def choose_best_rewrite_for_token(model: kenlm.Model, token: str, cfg: FixConfig) -> Optional[Tuple[str, Dict]]:
#     """
#     Return (rewritten_text, debug_info) if accepted, else None.
#     """
#     if not token.isalpha():
#         return None

#     if not (len(token) >= cfg.min_glued_token_len or should_try_short_glue(token, cfg)):
#         return None

#     base_avg, _, _ = kenlm_avg_score(model, token)

#     candidates: List[Tuple[str, float, str]] = []  # text, avg, tag

#     dp1 = dict_dp_segment(token, cfg)
#     if dp1:
#         cand = " ".join(dp1)
#         avg, _, _ = kenlm_avg_score(model, cand)
#         candidates.append((cand, avg, "dict_dp"))

#     dp2 = dict_dp_with_unknown_prefix(token, cfg)
#     if dp2:
#         cand = " ".join(dp2)
#         avg, _, _ = kenlm_avg_score(model, cand)
#         candidates.append((cand, avg, "dp_unknown_prefix"))

#     beam = kenlm_beam_segment(model, token, cfg)
#     if beam:
#         cand = " ".join(beam)
#         avg, _, _ = kenlm_avg_score(model, cand)
#         candidates.append((cand, avg, "kenlm_beam"))

#     if not candidates:
#         return None

#     candidates.sort(key=lambda x: x[1], reverse=True)
#     best_text, best_avg, best_tag = candidates[0]
#     delta = best_avg - base_avg

#     dbg = {
#         "token": token,
#         "base_avg": base_avg,
#         "best_avg": best_avg,
#         "delta": delta,
#         "best_tag": best_tag,
#         "candidates": candidates[:5],
#     }

#     if delta >= cfg.accept_delta_avg:
#         return best_text, dbg
#     return None
def _piece_known(piece: str) -> bool:
    lp = piece.lower()
    if zipf_frequency is not None:
        try:
            freq = zipf_frequency(lp, "en")  # type: ignore[arg-type]
        except Exception:
            freq = 0.0
        return freq >= 2.5
    return lp in COMMON_WORDS


def _candidate_stats(text: str) -> Tuple[int, int]:
    pieces = text.split()
    unknown = 0
    short = 0
    for p in pieces:
        lp = p.lower()
        if len(lp) <= 3:
            short += 1
        if not _piece_known(lp):
            unknown += 1
    return unknown, short


def choose_best_rewrite_for_token(model: kenlm.Model, token: str, cfg: FixConfig) -> Optional[Tuple[str, Dict]]:
    """
    Return (rewritten_text, debug_info) if accepted, else None.
    加强版：
      - 候选同时考虑 (unknown_count, short_piece_count, avg_score)
      - 对 dp_unknown_prefix 候选放宽一点阈值
    """
    if not token.isalpha():
        return None

    if token.lower() in COMMON_WORDS:
        return None

    special_prefix_match = re.match(r"^([A-Z][a-z]{2,12}?)([a-z].+)$", token) is not None

    if not (len(token) >= cfg.min_glued_token_len or should_try_short_glue(token, cfg) or special_prefix_match):
        return None

    base_avg, _, _ = kenlm_avg_score(model, token)

    raw_candidates: List[Tuple[str, float, str]] = []  # text, avg, tag

    dp1 = dict_dp_segment(token, cfg)
    if dp1:
        cand = " ".join(dp1)
        avg, _, _ = kenlm_avg_score(model, cand)
        raw_candidates.append((cand, avg, "dict_dp"))

    dp2 = dict_dp_with_unknown_prefix(token, cfg)
    if dp2:
        cand = " ".join(dp2)
        avg, _, _ = kenlm_avg_score(model, cand)
        raw_candidates.append((cand, avg, "dp_unknown_prefix"))

    beam = kenlm_beam_segment(model, token, cfg)
    if beam:
        cand = " ".join(beam)
        avg, _, _ = kenlm_avg_score(model, cand)
        raw_candidates.append((cand, avg, "kenlm_beam"))

    if not raw_candidates:
        return None

    # 计算每个候选的 unknown / short 片段数，并给一个综合 score
    cand_infos = []
    for cand_text, avg, tag in raw_candidates:
        unknown, short = _candidate_stats(cand_text)
        # 综合评分：avg - λ * unknown - μ * short
        # λ, μ 可以调。unknown 的惩罚更重一点。
        score = avg - 0.6 * unknown - 0.25 * short
        cand_infos.append((score, avg, unknown, short, tag, cand_text))

    # 综合分高的排前面
    cand_infos.sort(reverse=True, key=lambda x: x[0])
    best_score, best_avg, best_unknown, best_short, best_tag, best_text = cand_infos[0]

    delta = best_avg - base_avg

    dbg = {
        "token": token,
        "base_avg": base_avg,
        "best_avg": best_avg,
        "delta": delta,
        "best_tag": best_tag,
        "best_unknown": best_unknown,
        "best_short": best_short,
        "candidates": [(t, a, tag) for (_sc, a, _u, _s, tag, t) in cand_infos[:5]],
    }

    # 1) 对 dp_unknown_prefix（例如 Qwenmodelfamily）放宽一点要求：
    #    - 第一段未知（专有名词），后面的段都在 COMMON_WORDS -> 只要不比 base_avg 差太多就接受
    if best_tag == "dp_unknown_prefix":
        pieces = best_text.split()
        tail = [p.lower() for p in pieces[1:]]
        if tail and all(p in COMMON_WORDS for p in tail):
            # 允许略微不增益，只要没有明显更差
            if delta >= -0.001:
                return best_text, dbg
            # 否则继续走下面的通用规则

    if len(token) >= 30:
        pieces = best_text.split()
        if pieces:
            known = sum(1 for p in pieces if _piece_known(p))
            ratio_known = known / len(pieces)
            if ratio_known >= 0.5 and delta >= -0.003 and best_short <= 4:
                return best_text, dbg

    pieces = best_text.split()
    if pieces and all(_piece_known(p) for p in pieces):
        if delta >= -0.002:
            return best_text, dbg

    if delta >= cfg.accept_delta_avg:
        if best_unknown <= 2 and best_short <= 3:
            return best_text, dbg

    return None

# =============================
# Main fix routine
# =============================
def fix_text(model: kenlm.Model, text: str, cfg: FixConfig) -> str:
    t = normalize_unicode(text)
    t = normalize_hyphenation(t)
    t = normalize_whitespace(t)

    # 1) safe boundary spacing first (so Qwen3 separates from wepresentQwen3)
    t = conservative_boundary_spacing(t, cfg)

    # 2) protect model tokens/citations AFTER boundary spacing
    t, repl = protect_spans(t, cfg)

    # 3) mild punctuation cleanup
    t = mild_punct_cleanup(t)

    # 4) paragraph-level pass
    paragraphs = re.split(r"\n{2,}", t)
    out_paras: List[str] = []

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue

        suspicious = is_suspicious_paragraph(p, cfg)

        toks = tokenize(p)
        new_toks: List[str] = []

        for tok in toks:
            if tok.startswith("@@") and tok.endswith("@@"):
                new_toks.append(tok)
                continue

            base_token = None
            hyphen_candidate = False

            if tok.isalpha():
                base_token = tok
            else:
                if re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)+", tok):
                    base_token = tok.replace("-", "")
                    hyphen_candidate = True

            if base_token is not None:
                long_glue = len(base_token) >= cfg.min_glued_token_len
                do_try = long_glue or should_try_short_glue(base_token, cfg)
                if do_try:
                    res = choose_best_rewrite_for_token(model, base_token, cfg)
                    if res:
                        rewritten, dbg = res
                        if hyphen_candidate:
                            rewritten = _reinject_hyphen(tok, base_token, rewritten)
                        if cfg.debug:
                            print("\n[DEBUG] token:", dbg["token"])
                            print("  base_avg:", dbg["base_avg"])
                            print("  best_avg:", dbg["best_avg"], "delta:", dbg["delta"], "via", dbg["best_tag"])
                            for cand, av, tag in dbg["candidates"]:
                                print(f"   - {tag:16s} avg={av: .4f} : {cand}")
                        new_toks.extend(rewritten.split())
                    else:
                        new_toks.append(tok)
                else:
                    new_toks.append(tok)
            else:
                new_toks.append(tok)

        out_paras.append(detokenize(new_toks))

    out = "\n\n".join(out_paras)

    # 5) restore protected spans
    out = restore_spans(out, repl)

    out = re.sub(r"(\d),\s+(\d{3}\b)", r"\1,\2", out)

    out = re.sub(r"\b([A-Za-z]{3,})\s*-\s*of\s*-\s*([A-Za-z]{3,})\b", r"\1-of-\2", out)
    out = re.sub(r"\b([A-Za-z]{2,})\s*-\s*([A-Za-z]{2,})\b", r"\1-\2", out)
    out = re.sub(r"\b([A-Za-z]{2,10})\s*-\s*(\d+[A-Za-z]?)\b", r"\1-\2", out)
    out = re.sub(r"\b([A-Za-z]{2,10}-\d+)\s+([A-Za-z])\b", r"\1\2", out)
    out = re.sub(r"\b([A-Z][a-z]?)\s+([A-Z])\s*-\s*(\d+)\s*([A-Za-z])\b", r"\1\2-\3\4", out)
    out = re.sub(r"\(Mo\s+E\)", "(MoE)", out)
    out = re.sub(r"\b([A-Za-z]{2,}\d+)([A-Za-z]{3,})\b", r"\1 \2", out)
    out = re.sub(r"\b(\d+)(to)(\d+)\b", r"\1 \2 \3", out)
    out = re.sub(r"\b(from|to|around|about|over|under)(\d+)\b", r"\1 \2", out)
    out = re.sub(r"\b(\d{1,4})(languages?|dialects?)\b", r"\1 \2", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to KenLM .bin/.arpa.bin")
    ap.add_argument("--text", required=True, help="Input text")
    ap.add_argument("--delta", type=float, default=0.0015, help="Accept delta avg threshold")
    ap.add_argument("--beam", type=int, default=48, help="Beam size")
    ap.add_argument("--debug", action="store_true", help="Print debug info for rewritten tokens")
    args = ap.parse_args()

    cfg = FixConfig(accept_delta_avg=args.delta, beam_size=args.beam, debug=args.debug)
    m = kenlm.Model(args.model)

    fixed = fix_text(m, args.text, cfg)

    print("\n================ ORIGINAL ================\n")
    print(args.text)
    print("\n================= FIXED ==================\n")
    print(fixed)
    print("\n=========================================\n")


if __name__ == "__main__":
    main()



#python scripts/fix_glued_text_with_kenlm_v3.py   --model data/wiki_en_dep.arpa.bin   --debug   --text "In this work, wepresentQwen3, thelatestversionoftheQwenmodelfamily . Qwen3 comprises a series of large language models( LLMs) designed to advance performance, efficiency, and multilingual capabilities . TheQwen3seriesincludesmodelsofbothdense and Mixture - of - Expert( MoE) architectures, with parameter scales ranging from . to 235 billion . A key innovation in Qwen3 is the integration of thinking mode( for complex, multi - stepreasoning) andnon - thinkingmode( for rapid, context - drivenresponses) intoa unifiedframework . Thiseliminatestheneedtoswitchbetweendifferentmodels - - such as chat - optimized models( e . g ., GPT - 4o) and dedicated reasoning models( e . g ., QwQ - 32B) - - andenablesdynamicmodeswitchingbasedonuserqueriesorchattemplates ."
