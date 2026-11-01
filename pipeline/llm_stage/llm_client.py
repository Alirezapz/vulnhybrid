"""
Provider-agnostic LLM client for Stage 2. Anthropic and OpenAI are the two
hosted paths; "ollama" (or "openai" with a custom --base-url) targets any
locally-hosted OpenAI-compatible server -- Ollama, LM Studio, vLLM,
llama.cpp's server mode -- so this also works with a free local model and
no API key, at the cost of running on your own CPU/GPU instead of a
provider's.

IMPORTANT, stated plainly: the actual live API call in this module was
NOT validated against a real hosted model from the environment this was
built in -- no API credentials were available there (checked directly: a
bare request to api.anthropic.com came back 401, no key in the
environment). The local-model path's HTTP/JSON handling WAS validated end
-to-end against a real local mock server speaking the same protocol (see
test_llm_client.py) -- that confirms the networking and parsing logic, but
not how well any *specific* small model actually reasons about the task.
Everything else -- prompt construction, the tool schema, retry behavior on
malformed/failed output -- was tested against mocked provider functions.
Run a small --limit smoke test with your own API key or local model before
trusting this at scale.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

from .llm_schema import Verdict
from .prompts import SYSTEM_PROMPT, build_user_prompt, REPORT_TOOL_SCHEMA

logger = logging.getLogger("llm_client")

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


@dataclass
class LLMConfig:
    provider: str = "anthropic"          # "anthropic" | "openai" | "ollama"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0
    base_url: str = None     # override for OpenAI-compatible local servers (Ollama, LM Studio, vLLM, ...)
    api_key: str = None      # override; local servers usually ignore the actual value


def _call_anthropic(system: str, user: str, tool_schema: dict, config: LLMConfig):
    import anthropic  # local import: optional dependency
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=system,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": tool_schema["name"]},
        messages=[{"role": "user", "content": user}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_schema["name"]:
            return block.input, response.usage.input_tokens, response.usage.output_tokens
    raise ValueError("Model did not call the expected tool")


def _is_local_endpoint(config: LLMConfig) -> bool:
    """True when the target is a model on this machine (Ollama or a localhost
    base_url). Such a connection must not be sent through an external proxy."""
    if config.provider == "ollama":
        return True
    if not config.base_url:
        return False
    from urllib.parse import urlparse
    host = (urlparse(config.base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local")


def _call_openai_compatible(system: str, user: str, tool_schema: dict, config: LLMConfig):
    """Works against real OpenAI AND any locally-hosted OpenAI-compatible
    server (Ollama, LM Studio, vLLM, llama.cpp's server mode) -- they all
    implement the same /chat/completions shape. config.base_url=None means
    "use OpenAI's real endpoint"; set it to point at a local server
    instead (see _call_ollama below for the Ollama-specific preset).
    """
    import os
    import re
    import openai  # local import: optional dependency

    client_kwargs = {"api_key": config.api_key or os.environ.get("OPENAI_API_KEY", "not-needed-for-local")}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    # A LOCAL model must NOT be routed through an external HTTP/SOCKS proxy. A
    # machine-wide proxy (e.g. ALL_PROXY=socks://127.0.0.1:... , common where
    # the internet is only reachable via a proxy) otherwise hijacks even the
    # loopback connection to Ollama, and every call fails with "Unknown scheme
    # for proxy URL" (SOCKS support needs the httpx[socks] extra). Bypass env
    # proxies for local endpoints; keep them for hosted ones, where the user
    # may genuinely need the proxy to reach the API.
    if _is_local_endpoint(config):
        try:
            import httpx
            client_kwargs["http_client"] = httpx.Client(trust_env=False)
        except Exception:
            pass

    client = openai.OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=[{
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema["description"],
                "parameters": tool_schema["input_schema"],
            },
        }],
        tool_choice={"type": "function", "function": {"name": tool_schema["name"]}},
    )
    msg = response.choices[0].message
    usage = response.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0

    if msg.tool_calls:
        args = json.loads(msg.tool_calls[0].function.arguments)
        return args, in_tok, out_tok

    # Fallback for local models: smaller/quantized models served via Ollama
    # etc. don't always honor tool_choice as reliably as Claude/GPT-4 do --
    # try to recover a JSON object from the plain-text response before
    # giving up. This is exactly the kind of failure get_verdict's retry
    # loop is meant to absorb if it happens occasionally; this fallback
    # just reduces how OFTEN it has to.
    #
    # Two shapes the model may produce instead of a proper tool call:
    #   Shape A (flat):    {"is_vulnerable": ..., "confidence": ..., ...}
    #   Shape B (wrapped): {"name": "report_...", "arguments": {"is_vulnerable": ...}}
    # Shape B is what qwen2.5-coder:7b produces -- it copies the tool-call
    # wire format into plain text instead of using the actual tool_calls field.
    # We unwrap it so args_to_verdict gets the flat dict it expects.
    if msg.content:
        # Find the outermost {...} block in the response
        match = re.search(r"\{.*\}", msg.content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                # Unwrap Shape B: {"name": "...", "arguments": {...}}
                if isinstance(parsed, dict) and "arguments" in parsed and isinstance(parsed["arguments"], dict):
                    parsed = parsed["arguments"]
                # Accept if it has at least the required fields
                if isinstance(parsed, dict) and "is_vulnerable" in parsed:
                    return parsed, in_tok, out_tok
            except json.JSONDecodeError:
                pass
        # Last resort: look for a JSON block inside markdown fences
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", msg.content, re.DOTALL)
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if isinstance(parsed, dict) and "arguments" in parsed and isinstance(parsed["arguments"], dict):
                    parsed = parsed["arguments"]
                if isinstance(parsed, dict) and "is_vulnerable" in parsed:
                    return parsed, in_tok, out_tok
            except json.JSONDecodeError:
                pass

    # Include a snippet of what the model actually returned, so a model that
    # ignores tool-calling entirely (some Ollama models do, despite tool_choice)
    # is diagnosable from the error field instead of an opaque "no JSON" message.
    snippet = (msg.content or "").strip().replace("\n", " ")[:300]
    raise ValueError(
        "Model did not call the expected tool and no JSON found in its response "
        f"text. Raw response (first 300 chars): {snippet!r}")


def _call_ollama(system: str, user: str, tool_schema: dict, config: LLMConfig):
    """Thin preset over _call_openai_compatible: Ollama exposes an
    OpenAI-compatible endpoint at /v1 by default. Respects an explicit
    --base-url/--api-key override if the person gave one (e.g. Ollama
    running on a different host/port), otherwise fills in Ollama's
    standard local defaults.
    """
    from dataclasses import replace
    effective = replace(
        config,
        base_url=config.base_url or "http://localhost:11434/v1",
        api_key=config.api_key or "ollama",  # Ollama ignores the value, just needs *a* string
    )
    return _call_openai_compatible(system, user, tool_schema, effective)


PROVIDER_FUNCS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai_compatible,
    "ollama": _call_ollama,
}


def args_to_verdict(sample_id: str, model: str, args: dict, static_rule_ids: List[str]) -> Verdict:
    # bool("false") is True. Reject malformed provider output instead of
    # silently converting a textual safe verdict into a vulnerability.
    if not isinstance(args, dict) or type(args.get("is_vulnerable")) is not bool:
        raise ValueError("is_vulnerable must be a JSON boolean")
    confidence = args.get("confidence")
    if (type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        raise ValueError("confidence must be a finite number in [0, 1]")
    return Verdict(
        sample_id=sample_id,
        model=model,
        is_vulnerable=args["is_vulnerable"],
        confidence=float(confidence),
        cwe=args.get("cwe"),
        severity=args.get("severity"),
        taint_source=args.get("taint_source"),
        explanation=args.get("explanation", ""),
        suggested_fix=args.get("suggested_fix", ""),
        static_findings_shown=static_rule_ids,
        raw_response=json.dumps(args, ensure_ascii=False),
    )


def get_verdict(sample_id: str, code: str, language: str,
                 rule_messages: List[Tuple[str, int, str]],
                 config: LLMConfig) -> Tuple[Verdict, int, int]:
    """Returns (Verdict, input_tokens, output_tokens). Retries on
    transient/parse failures with exponential backoff; gives up after
    MAX_RETRIES and returns a Verdict with `error` set rather than
    raising, so a batch run doesn't die on one bad sample."""
    user_prompt = build_user_prompt(code, language, rule_messages)
    static_rule_ids = [r for r, _, _ in rule_messages]
    call_fn = PROVIDER_FUNCS.get(config.provider)
    if call_fn is None:
        raise ValueError(f"Unknown provider: {config.provider}")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            args, in_tok, out_tok = call_fn(SYSTEM_PROMPT, user_prompt, REPORT_TOOL_SCHEMA, config)
            verdict = args_to_verdict(sample_id, config.model, args, static_rule_ids)
            return verdict, in_tok, out_tok
        except Exception as e:
            last_error = e
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, sample_id, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))

    return Verdict(
        sample_id=sample_id, model=config.model, is_vulnerable=None, confidence=None,
        static_findings_shown=static_rule_ids, error=str(last_error),
    ), 0, 0
