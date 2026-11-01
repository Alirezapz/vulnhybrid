"""Unit tests for the pure statistics in scripts/statistical_analysis.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import statistical_analysis as SA  # noqa: E402


def _f1(yt, yp):
    tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def test_mcnemar_all_a_correct_is_significant():
    yt = [1] * 10
    n10, n01, p = SA.mcnemar_exact(yt, [1] * 10, [0] * 10)  # A always right, B always wrong
    assert (n10, n01) == (10, 0)
    assert p < 0.01                     # 2/2^10 ≈ 0.002


def test_mcnemar_identical_classifiers():
    yt, a = [1, 0, 1, 0], [1, 0, 0, 0]
    n10, n01, p = SA.mcnemar_exact(yt, a, a)
    assert (n10, n01) == (0, 0)
    assert p == 1.0


def test_mcnemar_symmetric_disagreement():
    yt = [1, 1, 1, 1]
    n10, n01, p = SA.mcnemar_exact(yt, [1, 1, 0, 0], [0, 0, 1, 1])
    assert (n10, n01) == (2, 2)
    assert p == 1.0                     # perfectly balanced -> no evidence either way


def test_bootstrap_ci_perfect_classifier():
    yt, yp = [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]
    pt, lo, hi = SA.bootstrap_ci(yt, yp, _f1, n_boot=200, seed=1)
    assert pt == 1.0 and lo == 1.0 and hi == 1.0


def test_bootstrap_ci_bounds_are_ordered_and_valid():
    yt, yp = [1, 0] * 20, [1, 1] * 20   # always-positive classifier
    pt, lo, hi = SA.bootstrap_ci(yt, yp, _f1, n_boot=500, seed=2)
    assert 0.0 <= lo <= pt <= hi <= 1.0


def test_label_from_sample_id():
    assert SA.label_from_sample_id("juliet_CWE78_..._bad") == 1
    assert SA.label_from_sample_id("juliet_CWE114_..._goodG2B_good") == 0
    assert SA.label_from_sample_id("devign_12345") is None   # no derivable label


import analyze_verdicts as AV  # noqa: E402


def test_cwe_from_sample_id():
    assert AV.cwe_from_sample_id("juliet_CWE415_Double_Free__x_02_bad") == "CWE-415"
    assert AV.cwe_from_sample_id("juliet_CWE78_..._good") == "CWE-78"
    assert AV.cwe_from_sample_id("devign_999") is None


def test_ece_perfect_calibration_is_zero():
    # confidence 0.5, half correct -> accuracy 0.5 == mean confidence -> ECE 0
    ece, rows = AV.expected_calibration_error([0.5, 0.5], [1, 0], n_bins=10)
    assert abs(ece) < 1e-9


def test_ece_overconfident():
    # confidence 0.9 but only half correct -> |0.5 - 0.9| = 0.4
    ece, _ = AV.expected_calibration_error([0.9, 0.9], [1, 0], n_bins=10)
    assert abs(ece - 0.4) < 1e-9


def test_load_records_devign_label_map(tmp_path):
    """Devign ids carry no _bad/_good marker: without a label map every record
    is skipped; with one (from the samples file) they are all usable."""
    import json
    verdicts = tmp_path / "v.jsonl"
    verdicts.write_text(
        json.dumps({"sample_id": "devign_test_1", "is_vulnerable": True, "confidence": 0.9})
        + "\n"
        + json.dumps({"sample_id": "devign_test_2", "is_vulnerable": False, "confidence": 0.6})
        + "\n")
    recs, skipped = AV.load_records(str(verdicts))
    assert recs == [] and skipped == 2

    samples = tmp_path / "s.jsonl"
    samples.write_text(
        json.dumps({"sample_id": "devign_test_1", "label": 1}) + "\n"
        + json.dumps({"sample_id": "devign_test_2", "label": 0}) + "\n")
    label_map = AV.load_label_map(str(samples))
    recs, skipped = AV.load_records(str(verdicts), label_map)
    assert skipped == 0 and len(recs) == 2
    assert [(r["label"], r["pred"]) for r in recs] == [(1, 1), (0, 0)]
