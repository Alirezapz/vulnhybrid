"""
Common sample schema shared by every dataset loader in this pipeline
(SARD / Juliet, Devign, and any future source).

Design decision (see project discussion): all three loaders emit samples at
FUNCTION-level granularity with a binary label (1 = vulnerable, 0 = not
vulnerable). This is forced on us by Devign, which only ever existed at
function granularity (Zhou et al., NeurIPS 2019), but it also resolves the
classification-granularity gap noted in the project proposal: by extracting
individual bad()/good() functions out of Juliet rather than treating a whole
multi-function file as one sample, Juliet and Devign end up comparable in
the same evaluation harness. SARD test suites that ship in the same
manifest.xml format (Juliet is technically one such suite) reuse this same
extraction logic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


@dataclass
class Sample:
    sample_id: str            # globally unique, e.g. "juliet_CWE78_017_bad"
    dataset: str               # "juliet" | "devign" | "sard"
    language: str              # "c" | "cpp" | "java" | ...
    code: str                  # the extracted function (or function group)
    label: int                 # 1 = vulnerable, 0 = not vulnerable
    cwe: Optional[str] = None         # e.g. "CWE-78"
    cwe_name: Optional[str] = None    # e.g. "OS Command Injection"
    source_file: Optional[str] = None
    flaw_line: Optional[int] = None   # line number inside the ORIGINAL file
    split: Optional[str] = None       # "train" | "val" | "test" | None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_jsonl(samples: List[Sample], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Sample]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(Sample(**d))
    return samples


def summarize(samples: List[Sample], title: str = "") -> str:
    """Plain-text summary table (counts by dataset/label/cwe) for sanity
    checks and for pasting into the thesis report as a table."""
    from collections import Counter

    lines = []
    if title:
        lines.append(f"=== {title} ===")
    lines.append(f"Total samples: {len(samples)}")

    by_dataset = Counter(s.dataset for s in samples)
    by_label = Counter(s.label for s in samples)
    by_lang = Counter(s.language for s in samples)
    by_cwe = Counter(s.cwe for s in samples if s.cwe)
    by_split = Counter(s.split for s in samples if s.split)

    lines.append(f"By dataset : {dict(by_dataset)}")
    lines.append(f"By label   : {{vulnerable(1): {by_label.get(1, 0)}, "
                 f"safe(0): {by_label.get(0, 0)}}}")
    lines.append(f"By language: {dict(by_lang)}")
    if by_split:
        lines.append(f"By split   : {dict(by_split)}")
    lines.append(f"Distinct CWEs: {len(by_cwe)}")
    top_cwes = ", ".join(f"{k}({v})" for k, v in by_cwe.most_common(10))
    lines.append(f"Top CWEs   : {top_cwes}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(
        "schema.py is a library module (the Sample dataclass + JSONL "
        "helpers) -- it has nothing to run on its own. Use one of:\n"
        "  python juliet_loader.py --root <juliet_checkout>\n"
        "  python devign_loader.py --source huggingface\n"
        "  python build_dataset.py --juliet-root <juliet_checkout> --devign-source huggingface"
    )
