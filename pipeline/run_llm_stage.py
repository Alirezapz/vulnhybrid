"""
Phase 3 entry point: runs the LLM verification stage over a Phase 1
dataset, optionally using Phase 2's findings.jsonl as evidence shown to
the model. Writes verdicts.jsonl, joinable back by sample_id.

COST WARNING, read before running at scale: this calls a paid API once
per sample. The full combined.jsonl has 192,201 samples -- do NOT point
this at the whole thing without thinking about cost first. Use --limit
and --split to scope a run, and check the token totals this script prints
against your provider's CURRENT pricing (not reproduced here since it
changes and a stale number would be actively misleading -- look it up at
your provider's site).

Two scope modes (--mode):
  verify-only (default): only samples that already have >=1 static finding.
      Cheaper, narrower -- the system's recall ceiling becomes static
      analysis's recall ceiling, which Phase 2 measured to be genuinely low
      for several CWEs (e.g. ~5-10% for the CWE-121/122 mechanisms
      dangerous-copy doesn't cover). This is the literal shape of the
      "Paper 2"-style hybrid your proposal's related-work section critiques.
  all-samples: runs on every sample in scope regardless of static findings,
      with static evidence attached only where it exists. Doesn't inherit
      static analysis's recall ceiling, but costs proportionally more.

Usage:
    python run_llm_stage.py --samples data/combined.jsonl \\
        --findings data/findings.jsonl --split test --limit 50 \\
        --mode verify-only --out data/verdicts.jsonl
"""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from schema import Sample, read_jsonl as read_samples
from findings_schema import Finding, read_jsonl as read_findings
from llm_stage.llm_schema import Verdict, write_jsonl as write_verdicts
from llm_stage.llm_client import get_verdict, LLMConfig
from hybrid_scoring import SECURITY_RELEVANT_RULE_IDS
from benchmark_hygiene import sanitize_code
from llm_stage.prompts import SYSTEM_PROMPT, REPORT_TOOL_SCHEMA, build_user_prompt

logger = logging.getLogger("run_llm_stage")


def _build_rule_messages(findings: List[Finding]) -> List[Tuple[str, int, str]]:
    return [(f.rule_id, f.snippet_line or f.file_line or 0, f.message) for f in findings]


def request_fingerprint(sample, findings_by_sample, config, sanitize=True):
    code = sanitize_code(sample.code, juliet=True) if sanitize and sample.dataset == 'juliet' else sample.code
    rules = _build_rule_messages(findings_by_sample.get(sample.sample_id, []))
    payload = dict(system=SYSTEM_PROMPT, user=build_user_prompt(code, sample.language, rules),
                   schema=REPORT_TOOL_SCHEMA, model=config.model, provider=config.provider,
                   base_url=config.base_url, temperature=config.temperature,
                   max_tokens=config.max_tokens, protocol='label-blind-v1' if sanitize else 'legacy-v1')
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _process_one(sample: Sample, findings_by_sample: Dict[str, List[Finding]],
                  config: LLMConfig, delay: float, sanitize: bool = True) -> Tuple[Verdict, int, int]:
    """One sample end-to-end. Runs in a worker thread under --concurrency>1,
    so this must not touch any shared mutable state -- it only reads
    findings_by_sample (never writes) and returns its result for the main
    thread to accumulate, same pattern as the old sequential loop's body."""
    try:
        rule_messages = _build_rule_messages(findings_by_sample.get(sample.sample_id, []))
        code = sanitize_code(sample.code, juliet=True) if sanitize and sample.dataset == 'juliet' else sample.code
        verdict, in_tok, out_tok = get_verdict(sample.sample_id, code, sample.language,
                                                rule_messages, config)
    except Exception as e:  # get_verdict already catches its own call failures;
        # this is a last-resort net for anything else (e.g. a bug in this
        # function itself) so one bad sample can't take down the whole pool.
        logger.warning("Unexpected error processing %s: %s", sample.sample_id, e)
        verdict = Verdict(sample_id=sample.sample_id, model=config.model,
                           is_vulnerable=None, confidence=None,
                           static_findings_shown=[], error=str(e))
        in_tok = out_tok = 0
    verdict.extra['request_sha256'] = request_fingerprint(sample, findings_by_sample, config, sanitize)
    verdict.extra['benchmark_hints_removed'] = sanitize and sample.dataset == 'juliet'
    if delay:
        time.sleep(delay)
    return verdict, in_tok, out_tok


def select_samples(samples: List[Sample], findings_by_sample: Dict[str, List[Finding]],
                    mode: str, split: str, max_per_cwe: int, limit: int, seed: int) -> List[Sample]:
    pool = samples
    if split:
        pool = [s for s in pool if s.split == split]
    if mode == "verify-only":
        pool = [s for s in pool if findings_by_sample.get(s.sample_id)]

    if max_per_cwe:
        rng = random.Random(seed)
        by_cwe: Dict[str, List[Sample]] = defaultdict(list)
        for s in pool:
            by_cwe[s.cwe or "none"].append(s)
        capped = []
        for cwe, group in sorted(by_cwe.items()):
            group.sort(key=lambda s: s.sample_id)
            rng.shuffle(group)
            capped.extend(group[:max_per_cwe])
        pool = capped

    pool = sorted(pool, key=lambda s: s.sample_id)
    random.Random(seed).shuffle(pool)
    if limit:
        pool = pool[:limit]
    return pool


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--findings", default=None,
                     help="Phase 2 findings.jsonl. Optional, but required for --mode verify-only.")
    ap.add_argument("--mode", choices=["verify-only", "all-samples"], default="verify-only")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", ""],
                     help="Which split to run on. Default 'test' -- that's what Phase 5 "
                          "evaluation needs; running on train/val here is mostly for "
                          "prompt-iteration, not for the metrics you'll report.")
    ap.add_argument("--max-per-cwe", type=int, default=None,
                     help="Cap samples per CWE for a stratified, cost-bounded run")
    ap.add_argument("--limit", type=int, default=None, help="Hard cap on total samples (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument('--keep-benchmark-hints', action='store_true',
                    help='Legacy ablation only: keep Juliet comments and label-bearing names.')
    ap.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default="anthropic")
    ap.add_argument("--model", default=None,
                     help="Defaults: claude-sonnet-4-6 (anthropic), gpt-4o-mini (openai), "
                          "qwen2.5-coder:7b (ollama)")
    ap.add_argument("--base-url", default=None,
                     help="Override the API endpoint -- e.g. a different host/port for Ollama, "
                          "or LM Studio/vLLM's OpenAI-compatible endpoint. Leave unset for "
                          "Ollama's default (http://localhost:11434/v1) or real OpenAI/Anthropic.")
    ap.add_argument("--api-key", default=None,
                     help="Override the API key. Local servers usually ignore the actual "
                          "value but still need *something* non-empty.")
    ap.add_argument("--delay", type=float, default=0.0,
                     help="Seconds to sleep between calls (rate-limit safety margin). "
                          "Applied per-request even under --concurrency.")
    ap.add_argument("--concurrency", type=int, default=4,
                     help="Samples processed in parallel (default 4). These are I/O-bound "
                          "HTTP calls, so threads overlap the waiting instead of doing it "
                          "one at a time -- this is usually the single biggest lever on "
                          "total wall-clock time. Set to 1 for the old strictly-sequential "
                          "behavior. If you see a flood of connection errors that weren't "
                          "there before, your Ollama instance can't keep up -- lower this "
                          "rather than raising it further.")
    ap.add_argument("--include-nonsecurity-findings", action="store_true",
                     help="Also treat style/quality findings (variableScope, "
                          "duplicateExpression, ...) as static evidence for selection and "
                          "prompts. Off by default: only SECURITY_RELEVANT_RULE_IDS count.")
    ap.add_argument("--max-consecutive-errors", type=int, default=10,
                     help="Abort (exit code 2) after this many consecutive failed samples. "
                          "Catches a dead/unreachable model server within ~a minute instead "
                          "of burning the whole batch on retry loops (this exact failure "
                          "mode once turned a 1072-sample run into 45 minutes of pure "
                          "errors). 0 disables.")
    ap.add_argument("--out", default="data/verdicts.jsonl")
    args = ap.parse_args()

    if args.model is None:
        args.model = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini",
                       "ollama": "qwen2.5-coder:7b"}[args.provider]

    samples = read_samples(args.samples)
    findings_by_sample: Dict[str, List[Finding]] = defaultdict(list)
    if args.findings:
        n_total = n_kept = 0
        for f in read_findings(args.findings):
            n_total += 1
            # Phase 2's cppcheck --enable=all also emits style/quality checks
            # (variableScope, duplicateExpression, shadowVariable, ...). Those
            # are useful raw data but must NOT drive verify-only selection or
            # appear as "static evidence" in prompts -- hybrid_scoring already
            # ignores them at scoring time for the same reason, and this keeps
            # the LLM stage consistent with it. Confirmed live: without this
            # filter, samples whose only finding was 'variableScope' were being
            # selected for "verification" of a non-security style note.
            if not args.include_nonsecurity_findings \
                    and f.rule_id not in SECURITY_RELEVANT_RULE_IDS:
                continue
            n_kept += 1
            findings_by_sample[f.sample_id].append(f)
        if n_total != n_kept:
            logger.info("Filtered findings to %d security-relevant (dropped %d "
                        "style/quality findings; pass --include-nonsecurity-findings "
                        "to keep them).", n_kept, n_total - n_kept)
    elif args.mode == "verify-only":
        ap.error("--mode verify-only needs --findings (nothing to verify without it)")

    scope = select_samples(samples, findings_by_sample, args.mode, args.split,
                            args.max_per_cwe, args.limit, args.seed)
    logger.info("Selected %d samples (mode=%s, split=%r) out of %d total",
                len(scope), args.mode, args.split, len(samples))
    if not scope:
        logger.warning("Nothing selected -- check --mode/--split/--findings. Exiting.")
        return

    config = LLMConfig(provider=args.provider, model=args.model,
                        base_url=args.base_url, api_key=args.api_key)
    scope_by_id = {s.sample_id: s for s in scope}
    # Resume from checkpoint: skip samples already written to --out, AND
    # preload their Verdicts -- every checkpoint below overwrites --out with
    # the full in-memory `verdicts` list, so without preloading here, the
    # first checkpoint after a resume would silently drop everything saved
    # before the resume (confirmed by testing: run once, resume, and the
    # pre-resume entries vanish the moment the next checkpoint fires).
    already_done: set = set()
    existing_verdicts: List[Verdict] = []
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as _f:
            for _line in _f:
                if not _line.strip():
                    continue
                try:
                    d = json.loads(_line)
                    v = Verdict(**d)
                except Exception as exc:
                    raise SystemExit(f'Invalid checkpoint; refusing overwrite: {exc}')
                if v.sample_id in already_done:
                    raise SystemExit(f'Duplicate checkpoint ID: {v.sample_id}')
                sample = scope_by_id.get(v.sample_id)
                if sample is None or v.extra.get('request_sha256') != request_fingerprint(
                        sample, findings_by_sample, config, not args.keep_benchmark_hints):
                    raise SystemExit('Checkpoint model, prompt, input or protocol differs. Use a new --out path.')
                if v.error is None:
                    already_done.add(v.sample_id)
                    existing_verdicts.append(v)
        if already_done:
            logger.info("Resuming: %d samples already in %s, skipping them.",
                        len(already_done), args.out)

    scope = [s for s in scope if s.sample_id not in already_done]
    verdicts: List[Verdict] = existing_verdicts
    total_in_tok = total_out_tok = 0
    start_time = time.monotonic()
    CHECKPOINT_EVERY = 5  # was 10 -- more frequent so progress is visible sooner,
    # especially important now that concurrency means several samples can finish
    # in the same few seconds

    def _log_progress(i: int, total: int):
        elapsed = time.monotonic() - start_time
        rate = i / elapsed if elapsed > 0 else 0.0
        remaining = (total - i) / rate if rate > 0 else float("inf")
        logger.info(
            "Progress: %d/%d (%.1f/min). Tokens so far: %d in / %d out. "
            "Elapsed %s, ETA %s. Checkpointing...",
            i, total, rate * 60, total_in_tok, total_out_tok,
            _fmt_duration(elapsed), _fmt_duration(remaining) if remaining != float("inf") else "unknown",
        )

    if args.concurrency <= 1:
        # Simple sequential path, kept as an obviously-correct fallback and
        # for cases where a real (non-local) API needs strict request pacing.
        consecutive_errors = 0
        for i, s in enumerate(scope, 1):
            verdict, in_tok, out_tok = _process_one(s, findings_by_sample, config, args.delay,
                                                  not args.keep_benchmark_hints)
            verdicts.append(verdict)
            total_in_tok += in_tok
            total_out_tok += out_tok
            consecutive_errors = consecutive_errors + 1 if verdict.error else 0
            if args.max_consecutive_errors and consecutive_errors >= args.max_consecutive_errors:
                write_verdicts(verdicts, args.out)
                logger.critical(
                    "%d consecutive samples failed (last error: %s) -- the model "
                    "server is almost certainly down or unreachable. Progress so far "
                    "is checkpointed to %s; fix the server and re-run (resume will "
                    "skip what's already done).", consecutive_errors, verdict.error, args.out)
                sys.exit(2)
            if i % CHECKPOINT_EVERY == 0 or i == len(scope):
                _log_progress(i, len(scope))
                write_verdicts(verdicts, args.out)
    else:
        logger.info("Running with %d concurrent workers.", args.concurrency)
        completed = 0
        consecutive_errors = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_process_one, s, findings_by_sample, config, args.delay,
                                   not args.keep_benchmark_hints)
                       for s in scope]
            for future in as_completed(futures):
                verdict, in_tok, out_tok = future.result()
                verdicts.append(verdict)
                total_in_tok += in_tok
                total_out_tok += out_tok
                completed += 1
                consecutive_errors = consecutive_errors + 1 if verdict.error else 0
                if args.max_consecutive_errors and consecutive_errors >= args.max_consecutive_errors:
                    write_verdicts(verdicts, args.out)
                    logger.critical(
                        "%d consecutive samples failed (last error: %s) -- the model "
                        "server is almost certainly down or unreachable. Progress so far "
                        "is checkpointed to %s; fix the server and re-run (resume will "
                        "skip what's already done).", consecutive_errors, verdict.error, args.out)
                    pool.shutdown(wait=False, cancel_futures=True)
                    sys.exit(2)
                if completed % CHECKPOINT_EVERY == 0 or completed == len(scope):
                    _log_progress(completed, len(scope))
                    write_verdicts(verdicts, args.out)

    n_errors = sum(1 for v in verdicts if v.error)
    logger.info("Done. %d verdicts (%d errors) written to %s", len(verdicts), n_errors, args.out)
    logger.info("Total tokens: %d input / %d output -- check these against your provider's "
                "CURRENT pricing page before running a larger batch.", total_in_tok, total_out_tok)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


if __name__ == "__main__":
    main()
