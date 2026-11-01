"""
Schema for static-analysis tool output ("findings"), kept deliberately
separate from schema.py's Sample. A Sample is a labeled piece of ground
truth; a Finding is a tool's OPINION about a location in code. Phase 4
(hybrid scoring) needs to join these two by sample_id without conflating
"what the dataset says is true" with "what a tool claims it found" -- and
the LLM stage in Phase 3 will add a third, similarly-shaped opinion that
needs to join the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


@dataclass
class Finding:
    finding_id: str        # unique, e.g. "semgrep_<sample_id>_0"
    sample_id: str          # joins back to schema.Sample.sample_id
    tool: str                # "semgrep"
    rule_id: str             # e.g. "dangerous-copy"
    cwe: Optional[str] = None        # our own rule->CWE mapping, "CWE-120" style
    severity: Optional[str] = None   # tool-reported, e.g. "ERROR" | "WARNING" | "INFO"
    message: str = ""
    file_line: Optional[int] = None    # line number in the ORIGINAL file (audit/debug)
    snippet_line: Optional[int] = None  # line number within the extracted sample's own
                                          # code text (1-indexed) -- what the LLM stage
                                          # and any report will actually want to show
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_jsonl(findings: List[Finding], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for fnd in findings:
            f.write(json.dumps(fnd.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Finding]:
    findings = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            findings.append(Finding(**d))
    return findings


def group_by_sample(findings: List[Finding]) -> Dict[str, List[Finding]]:
    out: Dict[str, List[Finding]] = {}
    for fnd in findings:
        out.setdefault(fnd.sample_id, []).append(fnd)
    return out
