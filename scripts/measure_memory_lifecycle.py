"""
Measure recall / false-positive-rate of the memory-lifecycle detectors
(CWE-415 / 416 / 562) against Juliet ground truth, per CWE, per tool -- the
same methodology as Phase 2. Reproduces the table in
docs/RESULTS_memory_lifecycle.md.

Usage:
    python scripts/measure_memory_lifecycle.py <samples.jsonl> <findings.jsonl> [more_findings.jsonl ...]

`samples.jsonl` is Phase-1 output (juliet_loader.py) for the three CWE
directories; the findings file(s) are run_static_analysis.py output for
whichever tools you ran (pass several if you kept semgrep and cppcheck in
separate files).
"""
import json
import re
import sys
from collections import defaultdict

# Which (tool, rule_id) detectors are credited to each CWE.
DETECTORS = {
    "CWE-415": [("semgrep", "double-free"), ("cppcheck", "doubleFree")],
    "CWE-416": [("semgrep", "use-after-free"), ("cppcheck", "deallocuse"),
                ("cppcheck", "deallocret")],
    "CWE-562": [("semgrep", "return-stack-address"),
                ("cppcheck", "returnDanglingLifetime")],
}
_DIR_CWE = re.compile(r"CWE(\d+)_")


def load_samples(path):
    samples, n = {}, defaultdict(lambda: {0: 0, 1: 0})
    for line in open(path):
        d = json.loads(line)
        m = _DIR_CWE.search(d.get("extra", {}).get("file_path", "") or "")
        cwe = f"CWE-{m.group(1)}" if m else d.get("cwe")
        samples[d["sample_id"]] = (cwe, d["label"])
        n[cwe][d["label"]] += 1
    return samples, n


def load_hits(paths):
    hits = defaultdict(set)
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            hits[d["sample_id"]].add((d["tool"], d["rule_id"]))
    return hits


def rate(samples, n, cwe, detset):
    tp = fp = 0
    for sid, (c, label) in samples.items():
        if c != cwe:
            continue
        if hits.get(sid, set()) & detset:
            tp += label == 1
            fp += label == 0
    nv, ns = n[cwe][1], n[cwe][0]
    recall = tp / nv if nv else float("nan")
    fpr = fp / ns if ns else float("nan")
    lr = ((tp + 1) / (nv + 2)) / ((fp + 1) / (ns + 2)) if nv and ns else float("nan")
    return tp, fp, nv, ns, recall, fpr, lr


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    samples, n = load_samples(sys.argv[1])
    hits = load_hits(sys.argv[2:])

    print(f"{'CWE':9s} {'detector':38s} {'tp/nv':>10s} {'fp/ns':>10s} "
          f"{'recall':>7s} {'fp_rate':>8s} {'LR':>7s}")
    print("-" * 95)
    for cwe, dets in DETECTORS.items():
        for tool, rule in dets:
            tp, fp, nv, ns, rec, fpr, lr = rate(samples, n, cwe, {(tool, rule)})
            print(f"{cwe:9s} {tool + ':' + rule:38s} {f'{tp}/{nv}':>10s} "
                  f"{f'{fp}/{ns}':>10s} {rec:7.3f} {fpr:8.4f} {lr:7.1f}")
        tp, fp, nv, ns, rec, fpr, lr = rate(samples, n, cwe, set(dets))
        print(f"{cwe:9s} {'>> COMBINED (any detector)':38s} {f'{tp}/{nv}':>10s} "
              f"{f'{fp}/{ns}':>10s} {rec:7.3f} {fpr:8.4f} {lr:7.1f}\n")
