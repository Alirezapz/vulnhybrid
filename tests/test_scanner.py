"""
End-to-end scanner tests.

The static-analysis tests need semgrep on PATH (skipped if absent). The LLM
tests drive the *real* client against the pipeline's local mock OpenAI server
-- the same technique the thesis used to validate the HTTP/JSON path without a
paid API -- so they exercise triage -> verify -> hybrid-score -> status without
any network or key.
"""

import json
import os

import pytest

import vulnhybrid  # noqa: F401 (activates bootstrap)
from vulnhybrid import scanner as S
from vulnhybrid import report as R
from vulnhybrid.cli import _exit_code

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "examples", "vulnerable-code")

requires_semgrep = pytest.mark.skipif(
    not S.semgrep_available(), reason="semgrep not installed")


@pytest.fixture(scope="module")
def openai_installed():
    return pytest.importorskip("openai")


def _mock(port, tool_call):
    from llm_stage.mock_openai_server import start_mock_server
    return start_mock_server(port=port, respond_with_tool_call=tool_call)


def _llm(port):
    from llm_stage.llm_client import LLMConfig
    return LLMConfig(provider="openai", model="mock",
                     base_url=f"http://localhost:{port}/v1", api_key="x")


# ---------------------------------------------------------------------------
# Static-only
# ---------------------------------------------------------------------------

@requires_semgrep
def test_taint_flow_rules_produce_unconfirmed_candidates():
    """Taint-flow rules were measured near-non-discriminative (LR ~ 1), so a
    lone static hit stays a 'review' candidate -- never 'confirmed' -- in
    static mode. (Memory-lifecycle rules, measured 0-FP, behave differently:
    see the calibration test below.)"""
    rep = S.scan(EXAMPLES, mode="static")
    assert rep.files_scanned >= 4
    assert len(rep.candidates()) >= 5
    by_fn = {u.function: u for u in rep.units}
    for fn in ("run_user_report", "log_line", "greet"):
        assert fn in by_fn, fn
        assert by_fn[fn].status == "review"
        assert not by_fn[fn].predicted_vulnerable
    cwes = {u.cwe for u in rep.units}
    assert {"CWE-78", "CWE-120", "CWE-134"} <= cwes


@requires_semgrep
def test_memory_lifecycle_rules_fire_without_false_positives():
    rep = S.scan(os.path.join(EXAMPLES, "memory_lifecycle.c"), mode="static")
    rules_by_fn = {u.function: {f.rule_id for f in u.findings} for u in rep.units}
    # every planted bad_* function is caught with the right rule
    assert "double-free" in rules_by_fn.get("bad_double_free", set())
    assert "use-after-free" in rules_by_fn.get("bad_use_after_free_field", set())
    assert "use-after-free" in rules_by_fn.get("bad_use_after_free_deref", set())
    assert "return-stack-address" in rules_by_fn.get("bad_return_stack_buffer", set())
    assert "return-stack-address" in rules_by_fn.get("bad_return_local_address", set())
    # zero false positives: no memory-lifecycle rule fires on the safe counterparts
    mem_rules = {"double-free", "use-after-free", "return-stack-address"}
    for safe_fn in ("safe_realloc_then_free", "safe_use_before_free", "safe_return_heap"):
        assert not (rules_by_fn.get(safe_fn, set()) & mem_rules)
    cwes = {u.cwe for u in rep.units if u.cwe}
    assert {"CWE-415", "CWE-416", "CWE-562"} <= cwes


@requires_semgrep
def test_measured_memory_rules_confirm_on_their_own():
    """double-free / use-after-free were MEASURED on Juliet at ~0 false
    positives -> a high likelihood ratio -> a single static hit is enough to
    'confirm' even without the LLM. return-stack-address stayed unmeasured
    (0/3 on Juliet) and must remain a 'review' candidate."""
    rep = S.scan(os.path.join(EXAMPLES, "memory_lifecycle.c"), mode="static")
    by_fn = {u.function: u for u in rep.units}
    assert by_fn["bad_double_free"].predicted_vulnerable
    assert by_fn["bad_double_free"].status == "confirmed"
    assert by_fn["bad_use_after_free_field"].predicted_vulnerable
    assert by_fn["bad_use_after_free_deref"].predicted_vulnerable
    # unmeasured rule -> DEFAULT_STATS (LR ~ 1) -> stays a candidate
    assert by_fn["bad_return_stack_buffer"].status == "review"


@requires_semgrep
def test_single_file_scan():
    rep = S.scan(os.path.join(EXAMPLES, "command_injection.c"), mode="static")
    assert rep.files_scanned == 1
    assert {u.function for u in rep.units} == {"run_user_report", "run_fixed_report"}


@requires_semgrep
def test_exclude_glob():
    rep = S.scan(EXAMPLES, mode="static", excludes=("*command_injection*",))
    files = {u.file for u in rep.units}
    assert not any("command_injection" in f for f in files)


# ---------------------------------------------------------------------------
# Triage + LLM (mock)
# ---------------------------------------------------------------------------

@requires_semgrep
def test_triage_confirms_when_model_says_vulnerable(openai_installed):
    srv = _mock(11811, tool_call=True)
    try:
        rep = S.scan(EXAMPLES, mode="triage", llm=_llm(11811), concurrency=2)
    finally:
        srv.shutdown()
    # the mock says "vulnerable" for every unit it sees, so every function the
    # triage stage sent to the model comes back confirmed
    assert rep.llm_calls > 0 and rep.llm_errors == 0
    assert len(rep.flagged()) == rep.llm_calls
    assert all(u.status == "confirmed" for u in rep.flagged())
    # tokens are accumulated from the (mock) usage block
    assert rep.input_tokens > 0 and rep.output_tokens > 0


@requires_semgrep
def test_triage_clears_when_model_says_safe(openai_installed):
    # mock returns plain-text JSON (is_vulnerable=false) -> exercises the
    # OpenAI-compatible fallback parser AND the 'cleared' status
    srv = _mock(11812, tool_call=False)
    try:
        rep = S.scan(EXAMPLES, mode="triage", llm=_llm(11812), concurrency=2)
    finally:
        srv.shutdown()
    assert rep.llm_calls > 0 and rep.llm_errors == 0
    assert rep.flagged() == []
    assert len(rep.by_status("cleared")) == rep.llm_calls


@requires_semgrep
def test_max_units_caps_model_calls(openai_installed):
    srv = _mock(11813, tool_call=True)
    try:
        rep = S.scan(EXAMPLES, mode="triage", llm=_llm(11813), max_units=2)
    finally:
        srv.shutdown()
    assert rep.llm_calls == 2


@requires_semgrep
def test_boundary_verdict_is_uncertain_not_confirmed(monkeypatch):
    """A verdict of is_vulnerable=False with confidence 0.50 lands the score
    exactly on the threshold (llm_p = 1 - 0.5 = 0.5). That is a coin-flip, so
    the unit must read 'uncertain', never 'confirmed ... model: not vulnerable'."""
    from llm_stage.llm_schema import Verdict
    from llm_stage.llm_client import LLMConfig

    def fake_get_verdict(sample_id, code, language, rule_messages, config):
        return Verdict(sample_id=sample_id, model="fake", is_vulnerable=False,
                       confidence=0.5, severity="none", taint_source="bare param",
                       explanation="could not verify origin"), 10, 5

    monkeypatch.setattr(S, "get_verdict", fake_get_verdict)
    rep = S.scan(os.path.join(EXAMPLES, "command_injection.c"), mode="triage",
                 llm=LLMConfig(provider="openai", model="fake"), concurrency=1)
    assert rep.llm_errors == 0
    assert rep.uncertain(), "expected boundary verdicts to be 'uncertain'"
    assert not rep.by_status("confirmed")
    # every above-threshold unit is labelled uncertain, not confirmed
    for u in rep.units:
        if u.predicted_vulnerable:
            assert u.status == "uncertain"


# ---------------------------------------------------------------------------
# Report formats
# ---------------------------------------------------------------------------

@requires_semgrep
def test_json_report_is_valid_and_complete():
    rep = S.scan(EXAMPLES, mode="static")
    doc = json.loads(R.render_json(rep))
    assert doc["tool"] == "vulnhybrid"
    assert doc["summary"]["files_scanned"] == rep.files_scanned
    assert len(doc["results"]) == len(rep.units)
    assert all("status" in r and "score" in r for r in doc["results"])


@requires_semgrep
def test_sarif_report_is_valid():
    rep = S.scan(EXAMPLES, mode="static")
    doc = json.loads(R.render_sarif(rep))
    assert doc["version"] == "2.1.0"
    assert doc["runs"][0]["tool"]["driver"]["name"] == "vulnhybrid"
    # static-only review candidates are emitted as SARIF 'note'
    for res in doc["runs"][0]["results"]:
        assert res["level"] in ("error", "warning", "note")
        assert res["locations"][0]["physicalLocation"]["region"]["startLine"] >= 1


@requires_semgrep
def test_markdown_report_renders():
    rep = S.scan(EXAMPLES, mode="static")
    md = R.render_markdown(rep)
    assert md.startswith("# vulnhybrid scan report")
    assert "Candidates to review" in md


# ---------------------------------------------------------------------------
# CI exit codes
# ---------------------------------------------------------------------------

class _FakeReport:
    def __init__(self, confirmed=0, uncertain=0, review=0):
        self._c, self._u, self._r = confirmed, uncertain, review

    def by_status(self, status):
        return [object()] * {"confirmed": self._c, "uncertain": self._u,
                             "review": self._r}.get(status, 0)

    def uncertain(self):
        return [object()] * self._u

    def candidates(self):
        return [object()] * self._r


def test_exit_codes():
    assert _exit_code(_FakeReport(review=5), "none") == 0
    assert _exit_code(_FakeReport(review=5), "confirmed") == 0     # only review present
    assert _exit_code(_FakeReport(confirmed=2), "confirmed") == 1
    assert _exit_code(_FakeReport(uncertain=3), "confirmed") == 0  # hedged != confirmed
    assert _exit_code(_FakeReport(review=5), "review") == 1
    assert _exit_code(_FakeReport(uncertain=1), "review") == 1     # uncertain needs attention
    assert _exit_code(_FakeReport(), "any") == 0
    assert _exit_code(_FakeReport(review=1), "any") == 1
