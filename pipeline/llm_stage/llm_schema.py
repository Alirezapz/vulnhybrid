"""
Schema for Stage 2 (LLM semantic verification) output. A Verdict is the
LLM's opinion -- structurally the same ROLE as a Finding (Phase 2's static
analysis output), but kept as its own type because its fields are
genuinely different (confidence score, free-text explanation, suggested
fix) and Phase 4 needs to reason about "what did the static tool say" and
"what did the LLM say" as separate inputs before combining them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import os
import tempfile


@dataclass
class Verdict:
    sample_id: str               # joins back to schema.Sample.sample_id
    model: str                    # e.g. "claude-sonnet-4-6"
    is_vulnerable: Optional[bool]  # None if the call failed / couldn't parse
    confidence: Optional[float]    # 0.0-1.0, model's own self-reported confidence
    cwe: Optional[str] = None
    severity: Optional[str] = None       # "low" | "medium" | "high" | "critical"
    taint_source: Optional[str] = None    # where the model thinks tainted data
                                            # enters, "none identified", or null
    explanation: str = ""
    suggested_fix: str = ""
    static_findings_shown: List[str] = field(default_factory=list)  # rule_ids given as context
    raw_response: Optional[str] = None    # kept for audit/debugging, can be dropped to save space
    error: Optional[str] = None           # set if the call/parse failed
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_jsonl(verdicts: List[Verdict], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A stopped batch must not truncate its last successful checkpoint.
    fd, temporary = tempfile.mkstemp(dir=out_path.parent, prefix=out_path.name+'.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for v in verdicts:
                f.write(json.dumps(v.to_dict(), ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: str) -> List[Verdict]:
    verdicts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            verdicts.append(Verdict(**d))
    return verdicts


def by_sample(verdicts: List[Verdict]) -> Dict[str, Verdict]:
    """Reject ambiguous joins; select one model/run before evaluation."""
    result = {}
    for verdict in verdicts:
        if verdict.sample_id in result:
            raise ValueError(f"Duplicate verdict for {verdict.sample_id!r}; select one run")
        result[verdict.sample_id] = verdict
    return result
