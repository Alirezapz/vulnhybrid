"""
Prompt design for Stage 2 (LLM semantic verification).

This is built directly from what Phase 2 measured, not from generic
"ask an LLM to find bugs" advice. The concrete failure mode found in real
Juliet source -- a function Juliet itself labels SAFE calling the exact
same sink (`EXECL(...)`, `vsnprintf(dest, n, data, args)`) as the unsafe
version, differing only in whether `data` traces back to a hardcoded
string or external input a few lines earlier -- is a taint/data-provenance
question, not a pattern-matching question. So the prompt's central
instruction is: trace where the relevant value(s) actually came from
before deciding, don't just recognize the operation.

Deliberately does NOT tell the model which CWE to look for. Doing that
would leak the label and make the measured accuracy meaningless -- in a
real deployment you don't know the answer in advance either. A static
finding, when one exists, is presented as an unconfirmed lead from a
tool that's already been measured (Phase 2) to have real false-positive
rates -- worded that way on purpose, so the model doesn't anchor on it as
ground truth.
"""

SYSTEM_PROMPT = """You are a software security analyst reviewing a single function for vulnerabilities.

Your method, in order:
1. Identify any operation in this function that could be dangerous IF its inputs are attacker-influenced (e.g. a buffer write, a command execution, a format string, a path operation, a memory operation).
2. For each variable involved in that operation, trace it BACKWARD through this function: where was it assigned? Is it a hardcoded literal, a fixed constant, the result of a bounds-checked computation -- or does it come from a function parameter, external input, or a source you cannot fully verify from this code alone?
3. Only conclude the function is vulnerable if there is a plausible path from an untrusted or unverified source to a dangerous use. A dangerous-looking operation fed entirely by hardcoded or properly-validated values is NOT vulnerable, even if the operation itself looks identical to a vulnerable one elsewhere.
4. If you cannot see where a variable originates (e.g. it's a function parameter with no caller visible), say so explicitly and lower your confidence accordingly -- do not guess.
5. A static finding may point at a condition or style issue unrelated to the actual dangerous operation. Always read the ENTIRE function body and trace ALL variables that reach a dangerous sink -- do not stop your analysis at the flagged line.
6. For sinks that determine what code runs or which file/resource is accessed (library loading, process execution, dynamic file paths, deserialization, query construction) -- if the relevant variable is CONFIRMED to come from outside this function (a file read, environment variable, network input, command-line argument, or similar), treat that as sufficient grounds for is_vulnerable=true even without a visible attacker: the absence of a currently-provable attack path in this isolated function does not mean none exists elsewhere in the application. A bare function parameter with NO visible caller is NOT "confirmed external" for purposes of this rule -- that case still falls under Rule 4 (lower confidence, do not guess which way it goes). Only a hardcoded literal or an explicitly validated/allowlisted value counts as safe for these sink types.

A tool may have flagged a specific line as a possible concern. Treat that as an unconfirmed lead, not a confirmed fact -- the tool that produced it has a measured false-positive rate and routinely flags safe code whose sink call merely looks similar to a vulnerable one. Your job is to confirm or reject it using the reasoning above, not to defer to it.

Think through your reasoning, then call the `report_vulnerability_assessment` tool with your conclusion."""

USER_PROMPT_TEMPLATE = """Function under review (language: {language}):

```{language}
{code}
```
{static_evidence_block}
Analyze this function using the method described, then report your assessment."""

_STATIC_EVIDENCE_BLOCK_TEMPLATE = """
A static analysis tool flagged the following in this function (unconfirmed -- verify, don't assume):
{findings_text}
"""


def format_static_evidence(rule_messages: list) -> str:
    """rule_messages: list of (rule_id, line, message) tuples. Returns the
    block to splice into the user prompt, or "" if there's nothing to show
    (the model should still analyze the function from scratch in that case
    -- absence of a static finding is not evidence of safety, especially
    given Phase 2 measured low recall for several rule/CWE pairs)."""
    if not rule_messages:
        return ""
    lines = [f"  - line {line} ({rule_id}): {message}" for rule_id, line, message in rule_messages]
    return _STATIC_EVIDENCE_BLOCK_TEMPLATE.format(findings_text="\n".join(lines))


def build_user_prompt(code: str, language: str, rule_messages: list) -> str:
    return USER_PROMPT_TEMPLATE.format(
        language=language,
        code=code,
        static_evidence_block=format_static_evidence(rule_messages),
    )


# Tool schema for structured output via the model's native tool-calling --
# forces valid JSON shape rather than asking the model to "please return
# JSON" in plain text and hoping it complies.
REPORT_TOOL_SCHEMA = {
    "name": "report_vulnerability_assessment",
    "description": "Report the security assessment of the reviewed function.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_vulnerable": {
                "type": "boolean",
                "description": "True if there is a plausible untrusted-source-to-dangerous-sink path.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0. Lower this if you could not verify where a key "
                                "variable originates (e.g. it's a bare parameter).",
            },
            "cwe": {
                "type": ["string", "null"],
                "description": "Best-guess CWE id (e.g. 'CWE-78'), or null if not applicable/unsure.",
            },
            "severity": {
                "type": "string",
                "enum": ["none", "low", "medium", "high", "critical"],
            },
            "taint_source": {
                "type": "string",
                "description": "Where the relevant value(s) actually originate, in your own "
                                "words (e.g. 'hardcoded string literal', 'function parameter "
                                "with no visible caller', 'read from environment variable'). "
                                "This is the field most directly checked against the rest of "
                                "the function body's source-tracing reasoning.",
            },
            "explanation": {
                "type": "string",
                "description": "2-4 sentences: the operation in question, the trace, and the conclusion.",
            },
            "suggested_fix": {
                "type": "string",
                "description": "Concrete fix if vulnerable; empty string if not vulnerable.",
            },
        },
        "required": ["is_vulnerable", "confidence", "severity", "taint_source", "explanation"],
    },
}
