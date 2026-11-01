"""
Orchestrates the dataset pipeline: runs whichever loaders you point it at,
merges them into a single combined.jsonl, and prints a summary table you
can paste straight into the thesis report.

Examples:

    # Just Juliet, smoke-testing on the first 50 manifest entries
    python build_dataset.py --juliet-root juliet-test-suite-c --limit 50

    # Juliet + Devign via Hugging Face
    python build_dataset.py --juliet-root juliet-test-suite-c --devign-source huggingface

    # Juliet + Devign via the local-jsonl fallback
    python build_dataset.py --juliet-root juliet-test-suite-c \\
        --devign-source jsonl-dir --devign-jsonl-dir /path/to/codexglue/dataset

Either dataset can be omitted; the script reports what it actually built.
"""

from __future__ import annotations

import argparse
import logging

from schema import write_jsonl, summarize, Sample
from juliet_loader import load_juliet
from splitter import assign_grouped_splits
import devign_loader

logger = logging.getLogger("build_dataset")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--juliet-root", default=None,
                     help="Root of a Juliet checkout (skips Juliet if omitted)")
    ap.add_argument("--juliet-manifest", default=None)
    ap.add_argument("--devign-source", choices=["huggingface", "jsonl-dir", "none"],
                     default="none")
    ap.add_argument("--devign-jsonl-dir", default=None)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap Juliet manifest entries processed (smoke test)")
    ap.add_argument("--assign-splits", action="store_true",
                     help="Assign a CWE-stratified, leakage-safe train/val/test "
                          "split to Juliet samples (see splitter.py). Devign "
                          "already has official splits and is unaffected.")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    all_samples: list[Sample] = []

    if args.juliet_root:
        manifest = args.juliet_manifest or f"{args.juliet_root.rstrip('/')}/manifest.xml"
        juliet_samples = load_juliet(args.juliet_root, manifest, limit=args.limit)
        if args.assign_splits:
            assign_grouped_splits(
                juliet_samples,
                train=args.train_frac, val=args.val_frac, test=args.test_frac,
                seed=args.split_seed,
            )
        write_jsonl(juliet_samples, f"{args.out_dir}/juliet.jsonl")
        print(summarize(juliet_samples, "Juliet"))
        all_samples.extend(juliet_samples)
    else:
        logger.info("Skipping Juliet (--juliet-root not given)")

    if args.devign_source == "huggingface":
        devign_samples = devign_loader.load_devign_from_huggingface()
    elif args.devign_source == "jsonl-dir":
        if not args.devign_jsonl_dir:
            ap.error("--devign-jsonl-dir is required when --devign-source jsonl-dir")
        devign_samples = devign_loader.load_devign_from_jsonl_dir(args.devign_jsonl_dir)
    else:
        devign_samples = []
        logger.info("Skipping Devign (--devign-source not given)")

    if devign_samples:
        write_jsonl(devign_samples, f"{args.out_dir}/devign.jsonl")
        print(summarize(devign_samples, "Devign"))
        all_samples.extend(devign_samples)

    if not all_samples:
        ap.error("Nothing to do: pass --juliet-root and/or --devign-source")

    # Guard: sample_id is the join key for EVERYTHING downstream (findings
    # attribution, snippet-scan filenames, verdict matching, evaluation).
    # A silent duplicate here once collapsed all 27,318 Devign samples onto
    # 3 ids and quietly poisoned the w_llm fit, so this now fails loudly
    # instead of ever writing a corrupt combined.jsonl.
    from collections import Counter
    id_counts = Counter(s.sample_id for s in all_samples)
    dups = [(k, v) for k, v in id_counts.items() if v > 1]
    if dups:
        raise SystemExit(
            f"FATAL: {len(dups)} duplicate sample_id(s) in the combined dataset "
            f"(e.g. {dups[:3]}). Refusing to write combined.jsonl -- fix the "
            f"loader that produced them first."
        )
    logger.info("sample_id uniqueness check passed: %d samples, all unique.", len(all_samples))

    write_jsonl(all_samples, f"{args.out_dir}/combined.jsonl")
    print(summarize(all_samples, "Combined (everything)"))
    print(f"\nWrote {len(all_samples)} total samples to {args.out_dir}/combined.jsonl")


if __name__ == "__main__":
    main()
