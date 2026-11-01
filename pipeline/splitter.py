"""
Grouped, CWE-stratified train/val/test splitter for Juliet.

THE PROBLEM: Juliet generates many "flow variant" files per underlying
vulnerability -- e.g. CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket
shows up as _01.c, _02.c, ... _54c.c, _61a.c/_61b.c, ... -- each wrapping the
exact same flaw logic in a different superficial control/data-flow pattern.
A naive random per-sample split will scatter variants of the SAME underlying
case across train and test, so the model (or the LLM stage, or a learned
hybrid-scoring weight) can effectively memorize the pattern from a near-twin
it already saw in training. Reported metrics from that kind of split are
optimistic in a way that won't hold up once you're evaluating on genuinely
unseen code.

THE FIX: group all flow variants of the same base case together, and put
each whole group in exactly one split. Splitting decisions are made on
GROUPS, stratified by CWE, so rare CWEs aren't accidentally left out of a
split entirely just because of how the random shuffle landed.

Devign is untouched by this module because it ships CodeXGLUE splits. That
does NOT establish clone independence: audit code hashes and commits across
those splits separately (research/benchmark.py does so).
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from schema import Sample

logger = logging.getLogger("splitter")

# Strips a trailing flow-variant index like "_01", "_54c", "_22a" from a
# Juliet filename stem. Anchored at the END of the string on purpose --
# CWE numbers embedded mid-name (e.g. "...__CWE129_connect_socket...") must
# NOT be touched, only the trailing per-variant counter.
_VARIANT_SUFFIX_RE = re.compile(r"_\d{1,3}[a-z]?$")
_CWE_PREFIX_RE = re.compile(r"^(CWE\d+)")


def juliet_group_key(source_file: str) -> Tuple[str, str]:
    """(cwe_prefix, base_name), derived purely from the filename so that
    sibling files in a multi-file testcase (e.g. ..._22a.c / ..._22b.c)
    always land in the same group, even though only one of them carries a
    <flaw> tag in the manifest (and therefore only one of them has a
    non-None `Sample.cwe` -- which is why this does NOT key off
    `Sample.cwe` directly)."""
    stem = Path(source_file).stem
    base = _VARIANT_SUFFIX_RE.sub("", stem)
    m = _CWE_PREFIX_RE.match(stem)
    cwe_prefix = m.group(1) if m else "UNKNOWN"
    return (cwe_prefix, base)


def _allocate_with_floor(n: int, train: float, val: float, test: float) -> Tuple[int, int, int]:
    """Per-CWE group counts for (train, val, test).

    Plain `round(n * frac)` starves small categories: 61 of Juliet's 118
    CWEs have fewer than 10 base-case groups (some have exactly 1), so
    round(n * 0.1) rounds straight to 0 for most of them -- verified
    against the real manifest, not a hypothetical edge case. That leaves
    roughly half of all CWE categories with zero held-out representation,
    which would quietly undermine any later per-CWE results breakdown.

    This guarantees, where the group count physically allows it:
      n == 1  -> everything to train (nothing to hold out from a category
                 with only one base case anyway)
      n == 2  -> 1 train, 1 test (prioritize test over val when there's
                 only enough to guarantee one of them)
      n >= 3  -> at least 1 group in EACH of val and test
    For n >= ~10 this converges to the requested ratio; for n below that,
    expect visible deviation from 80/10/10 for that specific CWE -- this is
    a deliberate, logged trade-off, not an attempt to hide it.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_va = max(1, round(n * val))
    n_te = max(1, round(n * test))
    n_tr = n - n_va - n_te
    if n_tr < 1:
        # The floor on val+test ate the entire category. Shrink whichever
        # of the two is larger to free up at least one train group.
        if n_va >= n_te:
            n_va -= 1
        else:
            n_te -= 1
        n_tr = n - n_va - n_te
    return n_tr, n_va, n_te


def assign_grouped_splits(
    samples: List[Sample],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> None:
    """Mutates `sample.split` IN PLACE for every Juliet sample, using a
    CWE-stratified group split. Non-Juliet samples (e.g. Devign, which
    already has its own official split) are left untouched.

    Ratios are applied per-CWE, not globally, so a CWE with only a handful
    of base cases doesn't get statistically wiped out of val/test by one
    unlucky global shuffle -- though with very few groups for a given CWE,
    it's still possible for that CWE to end up with zero groups in val or
    test; see the warning this function logs for exactly which CWEs that
    happened to.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "ratios must sum to 1.0"
    rng = random.Random(seed)

    groups: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for s in samples:
        if s.dataset != "juliet" or not s.source_file:
            continue
        groups[juliet_group_key(s.source_file)].append(s)

    by_cwe: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for key in groups:
        by_cwe[key[0]].append(key)

    n_train_groups = n_val_groups = n_test_groups = 0
    starved_val, starved_test = [], []

    for cwe_prefix, keys in by_cwe.items():
        rng.shuffle(keys)
        n = len(keys)
        n_tr, n_va, n_te = _allocate_with_floor(n, train, val, test)
        train_keys = keys[:n_tr]
        val_keys = keys[n_tr:n_tr + n_va]
        test_keys = keys[n_tr + n_va:n_tr + n_va + n_te]

        if n > 0 and not val_keys:
            starved_val.append(cwe_prefix)
        if n > 0 and not test_keys:
            starved_test.append(cwe_prefix)

        n_train_groups += len(train_keys)
        n_val_groups += len(val_keys)
        n_test_groups += len(test_keys)

        for split_name, split_keys in (("train", train_keys), ("val", val_keys), ("test", test_keys)):
            for k in split_keys:
                for sample in groups[k]:
                    sample.split = split_name

    logger.info(
        "Grouped split across %d CWE strata, %d base-case groups total -> "
        "%d train / %d val / %d test groups",
        len(by_cwe), len(groups), n_train_groups, n_val_groups, n_test_groups,
    )
    if starved_val:
        logger.warning(
            "%d CWE(s) have ZERO groups in val (too few base cases to split "
            "further): %s", len(starved_val), starved_val,
        )
    if starved_test:
        logger.warning(
            "%d CWE(s) have ZERO groups in test (too few base cases to "
            "split further): %s", len(starved_test), starved_test,
        )

    verify_no_leakage(samples)


def verify_no_leakage(samples: List[Sample]) -> None:
    """Self-check: raises if any base-case group ended up spanning more
    than one split. This should be structurally impossible given how
    assign_grouped_splits assigns splits per-group, but a project like this
    is exactly the kind where it's worth asserting the invariant rather
    than just trusting the code that's supposed to enforce it."""
    group_splits: Dict[Tuple[str, str], set] = defaultdict(set)
    for s in samples:
        if s.dataset != "juliet" or not s.source_file or s.split is None:
            continue
        group_splits[juliet_group_key(s.source_file)].add(s.split)

    leaked = {k: v for k, v in group_splits.items() if len(v) > 1}
    if leaked:
        raise AssertionError(
            f"Split leakage detected in {len(leaked)} group(s) -- the same "
            f"base case appears in more than one split: "
            f"{list(leaked.items())[:5]}"
        )
    logger.info("Leakage self-check passed: every base-case group lives in exactly one split.")
