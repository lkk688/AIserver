#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix glued words (missing spaces) using:
1) punctuation + hyphen normalization
2) conservative boundary spacing (lower->Upper, letter<->digit)
3) dictionary DP segmentation (optional)
4) KenLM-scored beam-search segmentation (only for suspicious tokens)
5) conservative acceptance threshold

Designed for academic PDF extracted text cleanup.

Dependencies:
  pip install kenlm

Run:
  python scripts/fix_glued_text_with_kenlm.py --model data/wiki_en_dep.arpa.bin --text "<YOUR_TEXT>"

Tips:
- This is conservative: it will NOT rewrite heavily unless KenLM says it improves enough.
- Customize COMMON_WORDS with your domain vocabulary for better results.
"""

from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import kenlm


# -----------------------------
# Config
# -----------------------------
@dataclass
class FixConfig:
    # suspicious detection
    avg_token_len_threshold: float = 7.0
    long_token_len_threshold: int = 18
    long_token_ratio_threshold: float = 0.18

    # token segmentation triggers
    min_glued_token_len: int = 16

    # beam search
    beam_size: int = 24
    max_word_len: int = 24
    allow_unknown_words: bool = True
    unknown_word_penalty: float = 2.0  # penalty per unknown word piece
    word_count_penalty: float = 0.2    # encourages fewer splits

    # accept segmentation only if it improves LM avg score by >= threshold
    accept_delta_avg: float = 0.003

    # guardrails
    protect_model_like_tokens: bool = True
    protect_urls_emails: bool = True
    protect_citations: bool = True


# -----------------------------
# Minimal domain/common words
# (expand this for better DP/beam quality)
# -----------------------------
COMMON_WORDS = set("""
a an the and or of to in on for with from by as at into
this that these those it its we our you your they their
work introduce introduces introduced present presents
latest version model models family series includes comprise comprises
large language multilingual capabilities architecture architectures
dense mixture expert experts reasoning thinking mode modes
rapid context driven response responses unified framework
eliminates eliminate need switch between different such chat optimized
dedicated enables dynamic switching based user queries templates
mechanism allowing allocate computational resources adaptively during
inference balancing latency performance task complexity moreover leveraging
knowledge flagship significantly reduce required build smaller scale while
ensuring competitive empirical evaluations demonstrate achieves state art
results across diverse benchmarks including tasks code generation mathematical
agent competitive against larger proprietary compared predecessor expands
support languages dialects enhancing global accessibility improved cross lingual
understanding generation facilitate reproducibility community research development
publicly accessible under apache
""".split())


# -----------------------------
# Normalization helpers
# -----------------------------
PUNCT = r"""[,.;:!?()\[\]{}]"""

def normalize_unicode(text: str) -> str:
    # normalize weird spaces/dashes commonly found in PDF extraction
    text = text.replace("\u00a0", " ")  # NBSP
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return text

def normalize_hyphenation(text: str) -> str:
    # join "exam-\nple" -> "example" when letters around
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def punctuation_spacing(text: str) -> str:
    # Add spacing around punctuation, then clean up
    # Keep hyphen handled separately
    text = re.sub(f"({PUNCT})", r" \1 ", text)
    # remove space BEFORE common punctuation .,;:!? but keep after
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # normalize
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def conservative_boundary_spacing(text: str) -> str:
    # Insert spaces at safe boundaries:
    # lower->Upper: wepresentQwen -> wepresent Qwen
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # letter<->digit boundaries: Qwen3 -> Qwen 3 (sometimes desired, sometimes not)
    # We'll be conservative: only split if preceded by 3+ letters or followed by 2+ letters.
    text = re.sub(r"([A-Za-z]{3,})(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d+)([A-Za-z]{3,})", r"\1 \2", text)

    # punctuation directly followed by letter: "etc.,competitive" -> "etc., competitive"
    text = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", text)
    return text


# -----------------------------
# Protection: avoid breaking tokens like URLs, emails, model names
# -----------------------------
PROTECT_PATTERNS = [
    # URLs
    (re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE), "URL"),
    # Emails
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "EMAIL"),
    # citations-like "(OpenAI,2025)" or "(OpenAI, 2025)"
    (re.compile(r"\([A-Za-z][A-Za-z .-]*,\s*\d{4}[a-z]?\)"), "CITE"),
    # model-ish tokens: GPT-4o, Qwen3-235B-A22B, DeepSeek-R1, QwQ-32B
    (re.compile(r"\b[A-Za-z]{1,10}\d{1,4}(?:[-_][A-Za-z0-9]{1,12})+\b"), "MODEL"),
]

def protect_spans(text: str, cfg: FixConfig) -> Tuple[str, Dict[str, str]]:
    replacements: Dict[str, str] = {}
    idx = 0

    def should_apply(tag: str) -> bool:
        if tag in ("URL", "EMAIL") and not cfg.protect_urls_emails:
            return False
        if tag == "CITE" and not cfg.protect_citations:
            return False
        if tag == "MODEL" and not cfg.protect_model_like_tokens:
            return False
        return True

    for pat, tag in PROTECT_PATTERNS:
        if not should_apply(tag):
            continue

        def repl(m):
            nonlocal idx
            key = f"@@{tag}_{idx}@@"
            idx += 1
            replacements[key] = m.group(0)
            return key

        text = pat.sub(repl, text)

    return text, replacements

def restore_spans(text: str, replacements: Dict[str, str]) -> str:
    # restore in reverse order just in case
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text


# -----------------------------
# Suspicious paragraph detection
# -----------------------------
def is_suspicious_paragraph(text: str, cfg: FixConfig) -> bool:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return False
    lengths = [len(t) for t in tokens]
    avg_len = sum(lengths) / len(lengths)
    long_tokens = [l for l in lengths if l >= cfg.long_token_len_threshold]
    ratio = len(long_tokens) / max(len(lengths), 1)
    return (avg_len >= cfg.avg_token_len_threshold) or (ratio >= cfg.long_token_ratio_threshold)


# -----------------------------
# Tokenization (simple)
# We'll keep punctuation as separate tokens.
# -----------------------------
TOKEN_RE = re.compile(rf"[A-Za-z]+|[0-9]+|{PUNCT}|[-]+|@@[A-Z]+_\d+@@|[^\s]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)

def detokenize(tokens: List[str]) -> str:
    s = " ".join(tokens)
    # remove spaces before punctuation
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    # tighten parentheses/brackets spacing: "( word" -> "(word", "word )" -> "word)"
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    s = re.sub(r"\{\s+", "{", s)
    s = re.sub(r"\s+\}", "}", s)
    # normalize multi-spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Dictionary DP segmentation (covers full token only)
# -----------------------------
def dict_dp_segment(token: str, min_len: int = 16) -> Optional[List[str]]:
    if len(token) < min_len or not token.isalpha():
        return None
    lower = token.lower()
    n = len(lower)
    dp: List[Optional[List[str]]] = [None] * (n + 1)
    dp[0] = []
    for i in range(n):
        if dp[i] is None:
            continue
        for j in range(i + 1, min(n, i + 24) + 1):
            piece = lower[i:j]
            if piece in COMMON_WORDS:
                cand = dp[i] + [piece]
                if dp[j] is None or len(cand) < len(dp[j]):
                    dp[j] = cand
    if dp[n] is None or len(dp[n]) <= 1:
        return None

    # restore capitalization of first piece
    words = dp[n]
    if token[0].isupper():
        words[0] = words[0].capitalize()
    return words


# -----------------------------
# KenLM scoring utilities
# Use avg score (total / token_count) to compare fairly.
# -----------------------------
def kenlm_avg_score(model: kenlm.Model, text: str) -> Tuple[float, float, int]:
    fs = list(model.full_scores(text, bos=True, eos=True))
    n = max(len(fs), 1)
    total = model.score(text, bos=True, eos=True)
    return total / n, total, n


# -----------------------------
# KenLM beam-search segmentation for a single glued alphabetic token
# We score partial hypotheses by approximate LM avg score over the current words.
# -----------------------------
def kenlm_beam_segment(
    model: kenlm.Model,
    token: str,
    cfg: FixConfig
) -> Optional[List[str]]:
    if len(token) < cfg.min_glued_token_len or (not token.isalpha()):
        return None

    original_is_cap = token[0].isupper()
    t = token.lower()
    n = len(t)

    # Each beam state: (pos, words, score)
    # score = accumulated cost (lower is better)
    # We'll use negative avg-lm as a cost proxy + penalties.
    BeamState = Tuple[int, List[str], float]
    beam: List[BeamState] = [(0, [], 0.0)]

    def piece_cost(piece: str) -> float:
        # known words cheaper
        if piece in COMMON_WORDS:
            return 0.0
        return cfg.unknown_word_penalty if cfg.allow_unknown_words else 1e9

    for _step in range(n + 1):
        new_beam: List[BeamState] = []
        for pos, words, cost in beam:
            if pos >= n:
                new_beam.append((pos, words, cost))
                continue

            # try next slices
            max_j = min(n, pos + cfg.max_word_len)
            for j in range(pos + 1, max_j + 1):
                piece = t[pos:j]

                # basic pruning: avoid tiny fragments unless they are common (e.g., "a", "of")
                if len(piece) == 1 and piece not in COMMON_WORDS:
                    continue

                pc = piece_cost(piece)
                if pc >= 1e8:
                    continue

                cand_words = words + [piece]

                # LM score on current phrase (cheap)
                phrase = " ".join(cand_words)
                avg, _tot, _k = kenlm_avg_score(model, phrase)

                # convert "higher avg better" into a cost: (-avg)
                lm_cost = -avg

                # add word count penalty to discourage over-splitting
                wc_cost = cfg.word_count_penalty * len(cand_words)

                new_cost = lm_cost + wc_cost + pc + cost * 0.0  # keep deterministic

                new_beam.append((j, cand_words, new_cost))

        # keep top-K
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[: cfg.beam_size]

        # early stop if all complete
        if all(pos >= n for pos, _, _ in beam):
            break

    # pick best complete
    complete = [(w, c) for pos, w, c in beam if pos >= n]
    if not complete:
        return None
    complete.sort(key=lambda x: x[1])
    best_words = complete[0][0]

    # restore capitalization
    if original_is_cap and best_words:
        best_words[0] = best_words[0].capitalize()

    # require at least 2 words
    if len(best_words) <= 1:
        return None
    return best_words


# -----------------------------
# Decide whether to apply a segmentation based on LM delta
# -----------------------------
def choose_best_rewrite_for_token(
    model: kenlm.Model,
    token: str,
    cfg: FixConfig
) -> Optional[str]:
    """
    Return rewritten token (with spaces) if accepted, else None.
    Only for alphabetic tokens.
    """
    if not token.isalpha() or len(token) < cfg.min_glued_token_len:
        return None

    # baseline score on token alone
    base_avg, _, _ = kenlm_avg_score(model, token)

    candidates: List[Tuple[str, float]] = []

    # Candidate 1: dictionary DP
    dp_words = dict_dp_segment(token, min_len=cfg.min_glued_token_len)
    if dp_words:
        cand = " ".join(dp_words)
        avg, _, _ = kenlm_avg_score(model, cand)
        candidates.append((cand, avg))

    # Candidate 2: KenLM beam
    beam_words = kenlm_beam_segment(model, token, cfg)
    if beam_words:
        cand = " ".join(beam_words)
        avg, _, _ = kenlm_avg_score(model, cand)
        candidates.append((cand, avg))

    if not candidates:
        return None

    # best candidate by avg score
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_text, best_avg = candidates[0]

    # accept only if improvement is significant
    delta = best_avg - base_avg
    if delta >= cfg.accept_delta_avg:
        return best_text
    return None


# -----------------------------
# Main paragraph/document fix
# -----------------------------
def fix_text(model: kenlm.Model, text: str, cfg: FixConfig) -> str:
    # 0) unicode + hyphen + whitespace normalize
    text = normalize_unicode(text)
    text = normalize_hyphenation(text)
    text = normalize_whitespace(text)

    # 1) protect special spans
    protected, repl = protect_spans(text, cfg)

    # 2) conservative boundary spacing + punctuation spacing
    protected = conservative_boundary_spacing(protected)
    # Do not overdo punctuation spacing; do a mild pass
    protected = punctuation_spacing(protected)

    # 3) paragraph-level processing
    paragraphs = re.split(r"\n{2,}", protected)
    out_paras: List[str] = []

    for para in paragraphs:
        p = para.strip()
        if not p:
            continue

        # Only run segmentation if suspicious
        if not is_suspicious_paragraph(p, cfg):
            out_paras.append(p)
            continue

        toks = tokenize(p)
        new_toks: List[str] = []

        for tok in toks:
            # skip protected tokens placeholders
            if tok.startswith("@@") and tok.endswith("@@"):
                new_toks.append(tok)
                continue

            # only attempt for long alphabetic tokens
            if tok.isalpha() and len(tok) >= cfg.min_glued_token_len:
                rewritten = choose_best_rewrite_for_token(model, tok, cfg)
                if rewritten:
                    new_toks.extend(rewritten.split())
                else:
                    new_toks.append(tok)
            else:
                new_toks.append(tok)

        out_paras.append(detokenize(new_toks))

    result = "\n\n".join(out_paras)

    # 4) restore protected spans
    result = restore_spans(result, repl)

    # 5) final whitespace tidy
    result = normalize_whitespace(result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to KenLM .bin/.arpa.bin (e.g., data/wiki_en_dep.arpa.bin)")
    ap.add_argument("--text", required=True, help="Input text to fix (quote it)")
    ap.add_argument("--delta", type=float, default=0.003, help="Accept delta avg threshold (default 0.003)")
    ap.add_argument("--beam", type=int, default=24, help="Beam size (default 24)")
    args = ap.parse_args()

    cfg = FixConfig(accept_delta_avg=args.delta, beam_size=args.beam)

    model = kenlm.Model(args.model)
    fixed = fix_text(model, args.text, cfg)

    print("\n================ ORIGINAL ================\n")
    print(args.text)
    print("\n================= FIXED ==================\n")
    print(fixed)
    print("\n=========================================\n")


if __name__ == "__main__":
    main()