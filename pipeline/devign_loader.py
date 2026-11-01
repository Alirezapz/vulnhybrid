"""
Devign loader (27,318 manually-labeled C functions from FFmpeg and QEMU;
Zhou et al., "Devign," NeurIPS 2019). Devign itself never published official
train/val/test splits -- the de facto standard splits used in almost every
follow-up paper are the ones from Microsoft's CodeXGLUE benchmark
(80/10/10), distributed as `func` / `target` / `idx` per line.

Two ways to get the data -- try (1) first, fall back to (2) if it fails:

  1. Hugging Face `datasets` library (simplest, does the split for you):

        pip install datasets --break-system-packages
        python devign_loader.py --source huggingface --out data/devign.jsonl

     CAVEAT for this project: huggingface.co is a US-based service. If
     `load_dataset(...)` times out or refuses the connection from your
     network, that's most likely a regional access issue rather than a bug
     in this script -- try a VPN, or use path (2) below, which only needs
     Google Drive.

  2. Manual download via the original CodeXGLUE repo instructions
     (microsoft/CodeXGLUE, Code-Code/Defect-detection/README.md), which
     uses gdown against Google Drive and a small preprocessing script that
     emits train.jsonl / valid.jsonl / test.jsonl in the same func/target/idx
     shape as the HF version:

        pip install gdown --break-system-packages
        gdown https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
        # follow that repo's README to turn the raw download into
        # train.jsonl / valid.jsonl / test.jsonl, then:
        python devign_loader.py --source jsonl-dir --jsonl-dir /path/to/dataset --out data/devign.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from schema import Sample, write_jsonl, summarize

logger = logging.getLogger("devign_loader")

_HF_SPLIT_TO_OURS = {"train": "train", "validation": "val", "test": "test"}


def load_devign_from_huggingface() -> List[Sample]:
    from datasets import load_dataset  # local import: optional dependency

    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    samples: List[Sample] = []
    for hf_split, our_split in _HF_SPLIT_TO_OURS.items():
        if hf_split not in ds:
            continue
        for i, row in enumerate(ds[hf_split]):
            samples.append(_row_to_sample(row, our_split, fallback_idx=i))
    return samples


def load_devign_from_jsonl_dir(jsonl_dir: str) -> List[Sample]:
    """Reads CodeXGLUE's own preprocessed output: train.jsonl, valid.jsonl,
    test.jsonl in `jsonl_dir`, each line shaped {"func": ..., "target": 0|1,
    "idx": ...} -- the same schema the HF dataset card documents, so both
    paths converge on the same _row_to_sample() conversion."""
    file_to_split = {"train.jsonl": "train", "valid.jsonl": "val", "test.jsonl": "test"}
    samples: List[Sample] = []
    base = Path(jsonl_dir)
    found_any = False
    for fname, our_split in file_to_split.items():
        fpath = base / fname
        if not fpath.exists():
            logger.warning("Expected file not found: %s (skipping that split)", fpath)
            continue
        found_any = True
        with fpath.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(_row_to_sample(row, our_split, fallback_idx=i))
    if not found_any:
        raise FileNotFoundError(
            f"None of train.jsonl/valid.jsonl/test.jsonl found under {jsonl_dir}. "
            f"Run the CodeXGLUE Defect-detection preprocessing steps first."
        )
    return samples


def _row_to_sample(row: dict, split: str, fallback_idx: int) -> Sample:
    # Raw CodeXGLUE jsonl names this column "idx"; the Hugging Face mirror
    # renamed it to "id". The old code did row.get("idx") only, which on the
    # HF path silently returned None for EVERY row -- collapsing all 27,318
    # Devign samples onto three sample_ids (devign_train_None / _val_None /
    # _test_None). Confirmed live on the real dataset: one LLM verdict for
    # "devign_val_None" then matched 2,732 val samples with mixed labels and
    # poisoned the w_llm fit (the infamous n=2979 / recall=0.057 table).
    # Never trust a single column name again: try both, then fall back to an
    # enumeration index that is unique by construction.
    idx = row.get("idx")
    if idx is None:
        idx = row.get("id")
    if idx is None:
        idx = fallback_idx
    return Sample(
        sample_id=f"devign_{split}_{idx}",
        dataset="devign",
        language="c",
        code=row["func"],
        label=int(row["target"]),
        split=split,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["huggingface", "jsonl-dir"], default="huggingface")
    ap.add_argument("--jsonl-dir", default=None,
                     help="Required when --source jsonl-dir")
    ap.add_argument("--out", default="data/devign.jsonl")
    args = ap.parse_args()

    if args.source == "huggingface":
        try:
            samples = load_devign_from_huggingface()
        except ModuleNotFoundError:
            logger.error(
                "The 'datasets' package isn't installed. Run:\n"
                "    pip install datasets\n"
                "then re-run this command. (This is NOT a connectivity "
                "problem -- it's just a missing pip package.)"
            )
            raise
        except Exception as e:
            logger.error(
                "Hugging Face path failed (%s). If this looks like a "
                "network/connectivity error, retry with --source jsonl-dir "
                "after following the CodeXGLUE manual-download steps in "
                "this file's docstring.",
                e,
            )
            raise
    else:
        if not args.jsonl_dir:
            ap.error("--jsonl-dir is required when --source jsonl-dir")
        samples = load_devign_from_jsonl_dir(args.jsonl_dir)

    write_jsonl(samples, args.out)
    print(summarize(samples, "Devign"))
    print(f"\nWrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
