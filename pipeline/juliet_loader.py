"""
Juliet C/C++ loader.

Juliet is formally a SARD test suite, so the real work lives in
sard_manifest_parser.py -- this module just points it at a Juliet checkout
and is the natural place to add Juliet-specific quirks later (e.g. if you
add the Java portion of Juliet, which uses the same manifest.xml schema).

Getting the data (pick ONE):

  1. GitHub mirror (recommended -- this is what was used to verify the
     parser against real files during development):

        git clone --depth 1 https://github.com/arichardson/juliet-test-suite-c.git
        python juliet_loader.py --root juliet-test-suite-c --out data/juliet.jsonl

     This mirror ships manifest.xml at the repo root and source files under
     testcases/<CWE_DIR>/*.c, exactly the layout this loader expects.

  2. Official NIST SARD download (authoritative, but you do the unzip):
     https://samate.nist.gov/SARD/test-suites/112  (Juliet C/C++ 1.3)
     Unzip it, find the manifest.xml it ships with, then run as above
     pointing --root at the unzipped folder and --manifest at that file.

Either way, this script does NOT compile or run anything -- it only reads
source text and slices out functions, so the only prerequisite is the
checkout itself.
"""

from __future__ import annotations

import argparse
import logging

from sard_manifest_parser import parse_sard_testsuite
from schema import write_jsonl, summarize

logger = logging.getLogger("juliet_loader")


def load_juliet(root_dir: str, manifest_path: str, limit: int = None):
    return parse_sard_testsuite(
        root_dir=root_dir,
        manifest_path=manifest_path,
        dataset_name="juliet",
        default_language="c",
        limit=limit,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                     help="Root of the Juliet checkout (contains testcases/)")
    ap.add_argument("--manifest", default=None,
                     help="Path to manifest.xml (default: <root>/manifest.xml)")
    ap.add_argument("--out", default="data/juliet.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N manifest entries (smoke test)")
    args = ap.parse_args()

    manifest_path = args.manifest or f"{args.root.rstrip('/')}/manifest.xml"
    samples = load_juliet(args.root, manifest_path, limit=args.limit)
    write_jsonl(samples, args.out)
    print(summarize(samples, "Juliet"))
    print(f"\nWrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
