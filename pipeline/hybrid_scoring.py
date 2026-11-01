"""
Phase 4: combines Phase 2's static findings and Phase 3's LLM verdict into
one final score per sample.

THE CORE IDEA, and why it's not arbitrary weights: Phase 2 measured actual
(recall, false-positive-rate) pairs for every rule against real Juliet
ground truth. A rule's LIKELIHOOD RATIO -- P(rule fires | vulnerable) /
P(rule fires | safe), i.e. recall / fp_rate -- is exactly how much that
rule's firing should multiply your odds of "this is vulnerable", under
basic Bayesian reasoning. So instead of inventing trust weights, this
module computes them straight from the Phase 2 numbers:

  - dangerous-copy on CWE-121/122: recall ~= fp_rate -> likelihood ratio
    ~= 1.0 -> firing changes your belief almost NOT AT ALL. The math
    confirms what Phase 2 already showed empirically.
  - cppcheck arrayIndexOutOfBounds: ~0% measured FP rate -> a very large
    likelihood ratio -> firing should move your belief a lot, even though
    it fires rarely (low recall).

Multiple static findings on the same sample combine via sequential odds
multiplication (each rule's LR is conditionally independent evidence
relative to the same prior). The LLM verdict is combined with the static
evidence via a weighted average IN LOG-ODDS SPACE rather than naive
probability averaging or another LR multiplication -- the LLM's confidence
is already framed as its own probability estimate (a posterior, not a
pure likelihood), so multiplying it in as if it were a likelihood ratio
would double-count the prior. Log-odds weighted averaging is the standard,
simple way to combine two probability estimates without that error.

The default weight (w_llm=0.7) is a STARTING point, not a fitted value --
recommended next step once Phase 5 has real held-out results: fit w_llm
via a 1-variable logistic regression (or just a small grid search) against
validation-split ground truth, rather than trusting this default forever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from findings_schema import Finding
from llm_stage.llm_schema import Verdict

# (tool, rule_id, cwe_or_None) -> {"recall": ..., "fp_rate": ..., "note": ...}
# cwe=None entries are pooled/fallback values used when the sample's CWE
# wasn't one of the ones actually measured against in Phase 2.
RULE_STATS: Dict[Tuple[str, str, Optional[str]], dict] = {
    ("semgrep", "command-injection", None):
        {"recall": 0.600, "fp_rate": 0.600, "note": "measured vs CWE-78, Juliet"},
    ("semgrep", "format-string-non-literal", None):
        {"recall": 0.417, "fp_rate": 0.417, "note": "measured vs CWE-134, Juliet"},
    ("semgrep", "dangerous-copy", "CWE-121"):
        {"recall": 0.096, "fp_rate": 0.097, "note": "measured vs CWE-121, Juliet"},
    ("semgrep", "dangerous-copy", "CWE-122"):
        {"recall": 0.057, "fp_rate": 0.056, "note": "measured vs CWE-122, Juliet"},
    ("semgrep", "dangerous-copy", None):
        {"recall": 0.077, "fp_rate": 0.077, "note": "pooled average of the two measured CWEs"},
    ("cppcheck", "arrayIndexOutOfBounds", "CWE-121"):
        {"recall": 0.010, "fp_rate": 0.0006, "note": "0/1572 measured FPs; Laplace-smoothed, not literally 0"},
    ("cppcheck", "arrayIndexOutOfBounds", "CWE-122"):
        {"recall": 0.106, "fp_rate": 0.0006, "note": "0/1559 measured FPs; Laplace-smoothed, not literally 0"},
    ("cppcheck", "arrayIndexOutOfBounds", None):
        {"recall": 0.058, "fp_rate": 0.0006, "note": "pooled average of the two measured CWEs"},
    ("cppcheck", "nullPointer", None):
        {"recall": 0.10, "fp_rate": 0.10, "note": "measured vs CWE-121/122; noise-level, not discriminative"},
    ("cppcheck", "uninitvar", None):
        {"recall": 0.13, "fp_rate": 0.138, "note": "measured vs CWE-121; noise-level, not discriminative"},

    # --- Memory-lifecycle semgrep rules, MEASURED on Juliet ------------------
    # (see docs/RESULTS_memory_lifecycle.md). Same asymmetric shape as
    # cppcheck's arrayIndexOutOfBounds: low recall (Juliet spreads the flaw
    # across nested blocks / helper functions / separate files that an
    # intra-procedural syntactic rule can't reach) but a measured 0 false
    # positives on the synthetic good functions -> a large likelihood ratio,
    # so firing is strong (if rare) evidence rather than noise. fp_rate is
    # Laplace-smoothed off the measured 0, exactly as arrayIndexOutOfBounds is.
    ("semgrep", "double-free", "CWE-415"):
        {"recall": 0.0125, "fp_rate": 0.0006,
         "note": "measured vs CWE-415 Juliet: 12/962 recall, 0/1642 FP (Laplace-smoothed)"},
    ("semgrep", "double-free", None):
        {"recall": 0.0125, "fp_rate": 0.0006, "note": "same as CWE-415 measurement (rule is CWE-415-specific)"},
    ("semgrep", "use-after-free", "CWE-416"):
        {"recall": 0.0174, "fp_rate": 0.0019,
         "note": "measured vs CWE-416 Juliet: 8/459 recall, 0/519 FP (Laplace-smoothed)"},
    ("semgrep", "use-after-free", None):
        {"recall": 0.0174, "fp_rate": 0.0019, "note": "same as CWE-416 measurement (rule is CWE-416-specific)"},
    # semgrep return-stack-address (CWE-562) intentionally absent: 0/3 recall
    # on Juliet's only 3 CWE-562 cases (all route the address through a helper
    # function) -> stays DEFAULT_STATS.

    # cppcheck's DATAFLOW checkers on the same corpus -- the real winners on
    # memory-lifecycle, since value-flow tracks the flaw across the nested
    # blocks / helper calls / separate files that defeat a syntactic rule.
    # Same 0-FP shape, much higher recall. (See docs/RESULTS_memory_lifecycle.md.)
    ("cppcheck", "doubleFree", "CWE-415"):
        {"recall": 0.291, "fp_rate": 0.0006,
         "note": "measured vs CWE-415 Juliet: 280/962 recall, 0/1642 FP (Laplace-smoothed)"},
    ("cppcheck", "doubleFree", None):
        {"recall": 0.291, "fp_rate": 0.0006, "note": "same as CWE-415 measurement"},
    ("cppcheck", "deallocuse", "CWE-416"):
        {"recall": 0.031, "fp_rate": 0.0019,
         "note": "measured vs CWE-416 Juliet: 14/459 recall, 0/519 FP (Laplace-smoothed)"},
    ("cppcheck", "deallocuse", None):
        {"recall": 0.031, "fp_rate": 0.0019, "note": "same as CWE-416 measurement"},
    ("cppcheck", "returnDanglingLifetime", "CWE-562"):
        {"recall": 1.0, "fp_rate": 0.2,
         "note": "3/3 recall, 0/3 FP on Juliet -- but n=3 is TINY; heavily "
                 "Laplace-smoothed (LR ~ 5), treat as a weak positive signal"},
}

# Default for any (tool, rule_id) never measured against real ground truth.
# Deliberately close to "uninformative": a real measurement should replace
# this the moment one exists, rather than letting an unmeasured rule
# quietly act as if it were trustworthy.
DEFAULT_STATS = {"recall": 0.10, "fp_rate": 0.10, "note": "UNMEASURED -- treated as near-uninformative"}

# Pure style/cleanliness cppcheck checks (variableScope, constVariable, ...)
# never enter the score at all -- they're not security findings.
SECURITY_RELEVANT_RULE_IDS = {
    "dangerous-copy", "command-injection", "format-string-non-literal",
    "arrayIndexOutOfBounds", "bufferAccessOutOfBounds", "nullPointer",
    "uninitvar", "doubleFree", "useAfterFree", "memleak", "negativeIndex",
    # Memory-lifecycle detectors targeting the flaw families the LLM was
    # measured weakest on (CWE-415/416/562). The semgrep rules live in
    # static_analysis/rules/memory_safety.yml; the cppcheck ids are its native
    # dataflow checks. All were measured on Juliet (docs/RESULTS_memory_lifecycle.md)
    # -- double-free/use-after-free/doubleFree/deallocuse/returnDanglingLifetime
    # now carry real (0-FP, high-LR) weight via RULE_STATS; only semgrep's
    # return-stack-address remains on DEFAULT_STATS (0/3 on Juliet).
    "double-free", "use-after-free", "return-stack-address",
    "deallocuse", "returnDanglingLifetime",
}

MIN_FP_RATE = 0.0001   # floor, avoids divide-by-zero / infinite likelihood ratios --
# must stay BELOW every real measured fp_rate in RULE_STATS (was 0.001, which
# is higher than arrayIndexOutOfBounds/CWE-121's measured 0.0006 and was
# silently clipping that specific signal from its true LR=16.67 down to 10;
# confirmed by testing, see conversation)
MAX_LIKELIHOOD_RATIO = 50.0  # cap, so one rule can't single-handedly force the score to 0/1


def _lookup_stats(tool: str, rule_id: str, cwe: Optional[str]) -> dict:
    if (tool, rule_id, cwe) in RULE_STATS:
        return RULE_STATS[(tool, rule_id, cwe)]
    if (tool, rule_id, None) in RULE_STATS:
        return RULE_STATS[(tool, rule_id, None)]
    return DEFAULT_STATS


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def static_probability(findings: List[Finding], prior: float = 0.4) -> Tuple[float, List[str]]:
    """Combines every security-relevant finding for one sample into a
    single probability via sequential odds multiplication. Returns
    (probability, [explanation strings]) -- the explanation list is meant
    to go straight into a human-readable report.
    """
    if not math.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("prior must be finite and strictly between 0 and 1")
    log_odds = logit(prior)
    explanations = []
    seen_rules = set()
    for f in findings:
        if f.rule_id not in SECURITY_RELEVANT_RULE_IDS:
            continue
        # RULE_STATS measures a binary rule-fires event, not independent
        # evidence for every matching line. Repeated matches must not multiply
        # that event's likelihood ratio again.
        key = (f.tool, f.rule_id)
        if key in seen_rules:
            continue
        seen_rules.add(key)
        stats = _lookup_stats(f.tool, f.rule_id, f.cwe)
        fp_rate = max(stats["fp_rate"], MIN_FP_RATE)
        lr = min(stats["recall"] / fp_rate, MAX_LIKELIHOOD_RATIO)
        log_odds += math.log(max(lr, 1e-12))
        explanations.append(
            f"{f.tool}:{f.rule_id} fired (LR={lr:.2f}, {stats['note']})"
        )
    p = sigmoid(log_odds)
    return p, explanations


@dataclass
class HybridResult:
    sample_id: str
    score: float                  # final probability estimate, 0-1
    label: int                    # thresholded prediction, 0 or 1
    static_p: float
    llm_p: Optional[float]
    explanation: List[str]


def hybrid_score(sample_id: str, findings: List[Finding], verdict: Optional[Verdict],
                  prior: float = 0.4, w_llm: float = 0.7, threshold: float = 0.5) -> HybridResult:
    if not math.isfinite(w_llm) or not 0 <= w_llm <= 1:
        raise ValueError("w_llm must be finite and in [0, 1]")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    static_p, static_explain = static_probability(findings, prior)

    llm_p = None
    explanation = list(static_explain)
    if verdict is not None and verdict.error is None and verdict.confidence is not None:
        if (type(verdict.is_vulnerable) is not bool
                or not math.isfinite(verdict.confidence)
                or not 0 <= verdict.confidence <= 1):
            raise ValueError("Invalid LLM verdict: expected boolean and finite confidence in [0, 1]")
        llm_p = verdict.confidence if verdict.is_vulnerable else (1 - verdict.confidence)
        explanation.append(
            f"LLM ({verdict.model}) assessed is_vulnerable={verdict.is_vulnerable} "
            f"confidence={verdict.confidence:.2f} -- taint_source: {verdict.taint_source!r}"
        )
        final_logit = w_llm * logit(llm_p) + (1 - w_llm) * logit(static_p)
    else:
        # No usable LLM verdict (not run on this sample, or the call failed) --
        # fall back to static evidence alone rather than guessing.
        final_logit = logit(static_p)
        if verdict is not None and verdict.error:
            explanation.append(f"LLM call failed ({verdict.error}) -- using static evidence only")

    score = sigmoid(final_logit)
    return HybridResult(
        sample_id=sample_id, score=score, label=int(score >= threshold),
        static_p=static_p, llm_p=llm_p, explanation=explanation,
    )
