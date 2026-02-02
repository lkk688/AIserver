#!/usr/bin/env python3
import re
import argparse
import kenlm

PUNCT = r"""[,.;:!?()\[\]{}]"""

def normalize_spaces(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def variant_base(text: str) -> str:
    # Keep as-is but normalize whitespace a bit
    return normalize_spaces(text)

def variant_punct_spaced(text: str) -> str:
    # Space around punctuation, then normalize
    s = re.sub(f"({PUNCT})", r" \1 ", text)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)  # remove space before common punct
    return normalize_spaces(s)

def variant_light_fixes(text: str) -> str:
    """
    Light, conservative fixes:
    - insert space between lowercase->Uppercase boundaries: wepresentQwen3 -> wepresent Qwen3
    - insert space between letter->digit boundaries: from29to119 -> from29 to119 (still imperfect)
    - fix missing spaces after '.' or ',' when followed by a letter: etc .,competitive -> etc ., competitive
    """
    s = text
    # lower->Upper: abcDef -> abc Def
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # letter->digit and digit->letter boundaries (very conservative)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    # ensure a space after punctuation if directly followed by a letter
    s = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", s)
    return normalize_spaces(s)

def variant_more_aggressive(text: str) -> str:
    """
    More aggressive heuristics:
    - do light fixes
    - also break common glued function words patterns (very rough):
      e.g., thelatest -> the latest, intoa -> into a, andnon -> and non, Thiseliminates -> This eliminates
    This is NOT your final algorithm—only for scoring comparison.
    """
    s = variant_light_fixes(text)

    # A tiny set of common glue patterns (demo only)
    patterns = [
        (r"\bthelatest\b", "the latest"),
        (r"\bthelatestversion\b", "the latest version"),
        (r"\bwepresent\b", "we present"),
        (r"\binto a\b", "into a"),  # already spaced sometimes
        (r"\bintoa\b", "into a"),
        (r"\bandnon\b", "and non"),
        (r"\bThiseliminates\b", "This eliminates"),
        (r"\bThiseliminatestheneed\b", "This eliminates the need"),
        (r"\bdifferentmodels\b", "different models"),
        (r"\bmodewitching\b", "mode switching"),
        (r"\bmodeswitching\b", "mode switching"),
        (r"\bbyleveraging\b", "by leveraging"),
        (r"\ballQwen3modelsarepubliclyaccessible\b", "all Qwen3 models are publicly accessible"),
    ]
    for pat, rep in patterns:
        s = re.sub(pat, rep, s)

    # Space around punctuation for readability (consistent tokenization)
    s = variant_punct_spaced(s)
    return normalize_spaces(s)

def avg_score(model: kenlm.Model, s: str) -> tuple[float, int, float]:
    """
    Returns: (total_score, token_count, avg_score)
    token_count derived from full_scores length.
    """
    fs = list(model.full_scores(s, bos=True, eos=True))
    n = max(len(fs), 1)
    total = model.score(s, bos=True, eos=True)
    return total, n, total / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to KenLM .bin/.arpa.bin")
    ap.add_argument("--text", required=True, help="Input text to test (quoted)")
    args = ap.parse_args()

    m = kenlm.Model(args.model)

    variants = [
        ("A_base", variant_base(args.text)),
        ("B_punct_spaced", variant_punct_spaced(args.text)),
        ("C_light_fixes", variant_light_fixes(args.text)),
        ("D_more_aggressive", variant_more_aggressive(args.text)),
    ]

    results = []
    print("\n=== Variants (preview first 220 chars) ===")
    for name, v in variants:
        print(f"{name}: {v[:220]}{'...' if len(v)>220 else ''}")

    print("\n=== KenLM scores ===")
    for name, v in variants:
        total, ntok, avg = avg_score(m, v)
        results.append((avg, total, ntok, name, v))
        print(f"{name:18s}  avg={avg:10.4f}  total={total:10.2f}  tokens={ntok:5d}")

    results.sort(reverse=True, key=lambda x: x[0])

    print("\n=== Ranking by avg score (higher is better) ===")
    for rank, (avg, total, ntok, name, _) in enumerate(results, 1):
        print(f"{rank:2d}) {name:18s}  avg={avg:10.4f}  total={total:10.2f}  tokens={ntok:5d}")

    best = results[0]
    print("\nBEST by avg score:", best[3])
    print("\nBEST text (first 600 chars):\n", best[4][:600], ("..." if len(best[4])>600 else ""))

if __name__ == "__main__":
    main()