#!/usr/bin/env python3
"""
Download a KenLM model from Hugging Face and run a quick sanity test.

Usage:
  python download_test_kenlm.py --repo BramVanroy/kenlm_wikipedia_en --out ./kenlm_models
  python download_test_kenlm.py --repo <your_repo> --filename <optional_specific_file>
"""

import argparse
import os
import sys
from pathlib import Path

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def pick_model_file(repo_id: str, files: list[str]) -> str:
    """
    Prefer binary models for fast loading:
      - *.arpa.bin, *.bin
    Fallback:
      - *.arpa (requires build_binary to convert for best performance)
    """
    preferred = []
    for f in files:
        lf = f.lower()
        if lf.endswith(".arpa.bin") or lf.endswith(".bin"):
            preferred.append(f)
    if preferred:
        # prefer arpa.bin if present
        preferred.sort(key=lambda x: (0 if x.lower().endswith(".arpa.bin") else 1, len(x)))
        return preferred[0]

    arpas = [f for f in files if f.lower().endswith(".arpa")]
    if arpas:
        arpas.sort(key=len)
        return arpas[0]

    raise RuntimeError(
        f"No KenLM model file found in repo {repo_id}. "
        f"Expected a .bin/.arpa.bin or .arpa file, got: {files}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face repo_id, e.g. BramVanroy/kenlm_wikipedia_en")
    ap.add_argument("--out", default="./kenlm_download", help="Output directory")
    ap.add_argument("--filename", default=None, help="Optional: specific filename in the repo to download")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- deps ---
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except Exception as ex:
        eprint("Missing dependency: huggingface_hub. Install with: pip install -U huggingface_hub")
        raise

    # --- list files ---
    eprint(f"[1/3] Listing files in {args.repo} ...")
    files = list_repo_files(args.repo)
    eprint(f"Found {len(files)} files.")

    # --- choose model file ---
    if args.filename:
        if args.filename not in files:
            raise RuntimeError(f"--filename '{args.filename}' not found. Available: {files}")
        model_file = args.filename
    else:
        model_file = pick_model_file(args.repo, files)

    eprint(f"[2/3] Downloading model file: {model_file}")
    local_path = hf_hub_download(
        repo_id=args.repo,
        filename=model_file,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded: {local_path}")

    # --- test load + scoring ---
    eprint("[3/3] Testing KenLM load + scoring ...")
    try:
        import kenlm  # pip install https://github.com/kpu/kenlm OR pip install kenlm (platform-dependent)
    except Exception:
        eprint(
            "Missing dependency: kenlm.\n"
            "Install options:\n"
            "  - If pip works on your platform: pip install kenlm\n"
            "  - Otherwise compile from source: https://github.com/kpu/kenlm\n"
        )
        raise

    # If it's .arpa, kenlm.Model may still load it, but .bin is preferred.
    model = kenlm.Model(local_path)

    s1 = "This is a simple test sentence ."
    s2 = "This is a simple test sentence sentence sentence ."
    s3 = "Thissisasimpletestsentence."  # intentionally weird

    score1 = model.score(s1, bos=True, eos=True)
    score2 = model.score(s2, bos=True, eos=True)
    score3 = model.score(s3, bos=True, eos=True)

    print("\n=== Sanity scores (higher is better in KenLM; scores are log-probabilities) ===")
    print(f"s1: {score1:.3f} | {s1}")
    print(f"s2: {score2:.3f} | {s2}")
    print(f"s3: {score3:.3f} | {s3}")

    # Very simple sanity expectations:
    # - The repeated sentence should generally score worse than the normal sentence.
    # - The glued sentence should generally score worse than the normal sentence.
    ok = True
    if not (score1 > score2):
        ok = False
        eprint("WARN: Expected score(s1) > score(s2), but got the opposite.")
    if not (score1 > score3):
        ok = False
        eprint("WARN: Expected score(s1) > score(s3), but got the opposite.")

    if ok:
        print("\n✅ KenLM model loaded and sanity checks look OK.")
    else:
        print("\n⚠️ KenLM model loaded, but sanity comparisons did not match expectations.")
        print("This can happen with unusual models/tokenization. Still usable, but review your LM and preprocessing.")

if __name__ == "__main__":
    main()