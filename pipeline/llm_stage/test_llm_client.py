"""
Tests for everything in llm_client.py and prompts.py that does NOT require
a live API call -- retry behavior, response-to-Verdict conversion, and
prompt formatting. Run this after any change to llm_client.py/prompts.py;
it catches structural bugs before they'd otherwise only surface against a
real (costly) API call.

Run: python -m llm_stage.test_llm_client
"""

from .llm_client import get_verdict, args_to_verdict, LLMConfig, PROVIDER_FUNCS
from .prompts import build_user_prompt, format_static_evidence


def test_args_to_verdict_basic():
    args = {
        "is_vulnerable": True, "confidence": 0.85, "cwe": "CWE-78",
        "severity": "high", "taint_source": "read from environment variable",
        "explanation": "test", "suggested_fix": "validate input",
    }
    v = args_to_verdict("s1", "claude-sonnet-4-6", args, ["command-injection"])
    assert v.is_vulnerable is True
    assert v.confidence == 0.85
    assert v.cwe == "CWE-78"
    assert v.static_findings_shown == ["command-injection"]
    print("PASS: args_to_verdict_basic")


def test_get_verdict_succeeds_first_try(monkeypatch_provider):
    def fake_call(system, user, schema, config):
        return ({
            "is_vulnerable": False, "confidence": 0.9, "severity": "none",
            "taint_source": "hardcoded literal", "explanation": "safe",
        }, 100, 50)
    PROVIDER_FUNCS["_test"] = fake_call
    config = LLMConfig(provider="_test")
    verdict, in_tok, out_tok = get_verdict("s2", "int f(){return 0;}", "c", [], config)
    assert verdict.error is None
    assert verdict.is_vulnerable is False
    assert in_tok == 100 and out_tok == 50
    print("PASS: get_verdict_succeeds_first_try")


def test_get_verdict_retries_then_succeeds():
    calls = {"n": 0}

    def flaky_call(system, user, schema, config):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("simulated transient failure")
        return ({"is_vulnerable": True, "confidence": 0.7, "severity": "medium",
                  "taint_source": "unknown"}, 10, 10)

    PROVIDER_FUNCS["_test_flaky"] = flaky_call
    config = LLMConfig(provider="_test_flaky")
    verdict, _, _ = get_verdict("s3", "code", "c", [], config)
    assert calls["n"] == 3, f"expected exactly 3 attempts, got {calls['n']}"
    assert verdict.error is None
    assert verdict.is_vulnerable is True
    print("PASS: get_verdict_retries_then_succeeds (retried twice, succeeded on 3rd)")


def test_get_verdict_gives_up_after_max_retries():
    def always_fails(system, user, schema, config):
        raise ValueError("simulated permanent failure")

    PROVIDER_FUNCS["_test_dead"] = always_fails
    config = LLMConfig(provider="_test_dead")
    verdict, in_tok, out_tok = get_verdict("s4", "code", "c", [], config)
    assert verdict.error is not None
    assert verdict.is_vulnerable is None
    assert in_tok == 0 and out_tok == 0
    print("PASS: get_verdict_gives_up_after_max_retries (no exception raised)")


def test_prompt_includes_static_evidence_as_unconfirmed():
    prompt = build_user_prompt("void f(){}", "c", [("command-injection", 5, "exec call")])
    assert "unconfirmed" in prompt.lower()
    assert "line 5" in prompt
    assert "command-injection" in prompt
    print("PASS: prompt_includes_static_evidence_as_unconfirmed")


def test_prompt_with_no_static_evidence_omits_block():
    prompt = build_user_prompt("void f(){}", "c", [])
    assert "flagged" not in prompt.lower()
    print("PASS: prompt_with_no_static_evidence_omits_block")


if __name__ == "__main__":
    test_args_to_verdict_basic()
    test_get_verdict_succeeds_first_try(None)
    test_get_verdict_retries_then_succeeds()
    test_get_verdict_gives_up_after_max_retries()
    test_prompt_includes_static_evidence_as_unconfirmed()
    test_prompt_with_no_static_evidence_omits_block()
    print("\nAll llm_client/prompts tests passed.")
