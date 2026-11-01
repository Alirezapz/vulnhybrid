"""
Generic parser for SARD-format test suites (manifest.xml + a tree of source
files). Juliet C/C++ is itself one such SARD test suite, so this module is
shared infrastructure rather than Juliet-specific code.

Verified manifest.xml schema (checked directly against the Juliet 1.3 C/C++
mirror, not just documentation):

    <container>
      <testcase>
        <file path="CWE114_..._01.c">
          <flaw line="121" name="CWE-114: Process Control"/>
        </file>
      </testcase>
      <testcase>
        <file path="CWE114_..._22a.c">
          <flaw line="41" name="CWE-114: Process Control"/>
        </file>
        <file path="CWE114_..._22b.c"/>   <!-- helper file, no flaw -->
      </testcase>
    </container>

A <file> with no <flaw> child is either a pure helper file (no bad() or
good() content of its own -- gets skipped) or, far more commonly, a file
that still has its own good()/bad() pair but whose flaw lives in a sibling
file for that test case (multi-file flow variants).

Within a single .c/.cpp file, Juliet wraps the vulnerable function in
`#ifndef OMITBAD ... #endif` and the safe function(s) in
`#ifndef OMITGOOD ... #endif`. Both guards can appear MORE THAN ONCE per
file: once around the actual function definition, and again (as a 2-4 line
snippet) around the call site inside main(). We disambiguate by keeping only
blocks above a minimum line count -- verified against the real corpus, the
definition block is always far larger than the call-site snippet.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from schema import Sample

logger = logging.getLogger("sard_manifest_parser")

LANG_BY_EXT = {
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".java": "java",
}

# Matches the start of a preprocessor conditional block: #if / #ifdef / #ifndef
_COND_OPEN_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b")
_COND_CLOSE_RE = re.compile(r"^\s*#\s*endif\b")
_GUARD_RE_TEMPLATE = r"^\s*#\s*ifndef\s+{guard}\b"

MIN_BLOCK_LINES = 5  # filters out the main()-call-site snippets


def _read_text(path: Path) -> Optional[str]:
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    logger.warning("Could not read %s with utf-8 or latin-1", path)
    return None


def _extract_guarded_blocks(lines: List[str], guard: str) -> List[Tuple[int, int, str]]:
    """Find every `#ifndef {guard} ... #endif` region in `lines`, properly
    handling nested conditionals inside the region (e.g. an #ifdef _WIN32
    block inside the bad() function). Returns (start_line, end_line, text)
    tuples, 0-indexed, inclusive of the body but excluding the guard/#endif
    lines themselves.
    """
    guard_re = re.compile(_GUARD_RE_TEMPLATE.format(guard=re.escape(guard)))
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if guard_re.match(lines[i]):
            depth = 1
            body_start = i + 1
            j = body_start
            while j < n and depth > 0:
                if _COND_OPEN_RE.match(lines[j]):
                    depth += 1
                elif _COND_CLOSE_RE.match(lines[j]):
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            body_end = j  # index of the matching #endif (exclusive of body)
            body_text = "\n".join(lines[body_start:body_end])
            blocks.append((body_start, body_end, body_text))
            i = body_end + 1
        else:
            i += 1
    return blocks


def extract_bad_and_good(
    file_text: str,
) -> Tuple[Optional[str], Optional[str], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (bad_code, good_code, bad_line_ranges, good_line_ranges).

    Line ranges are 1-indexed [start, end] INCLUSIVE pairs in the ORIGINAL
    file's own line numbering -- the same convention manifest.xml's
    <flaw line="..."/> uses. Phase 2 needs these to attribute a static
    analysis tool's file:line findings back to the correct extracted
    sample, since the tool runs against the original (compilable) file,
    not the isolated snippet.
    """
    lines = file_text.splitlines()

    bad_raw = [(s, e, t) for (s, e, t) in _extract_guarded_blocks(lines, "OMITBAD")
               if t.count("\n") + 1 >= MIN_BLOCK_LINES]
    good_raw = [(s, e, t) for (s, e, t) in _extract_guarded_blocks(lines, "OMITGOOD")
                if t.count("\n") + 1 >= MIN_BLOCK_LINES]

    bad_code = "\n\n".join(t for (_, _, t) in bad_raw) if bad_raw else None
    good_code = "\n\n".join(t for (_, _, t) in good_raw) if good_raw else None
    # body_start/body_end from _extract_guarded_blocks are 0-indexed,
    # body_end exclusive -> +1/+0 converts to 1-indexed inclusive.
    bad_ranges = [(s + 1, e) for (s, e, _) in bad_raw]
    good_ranges = [(s + 1, e) for (s, e, _) in good_raw]
    return bad_code, good_code, bad_ranges, good_ranges


def _fallback_line_window(file_text: str, flaw_line: int, window: int = 15) -> Tuple[str, Tuple[int, int]]:
    """Used only if the OMITBAD heuristic fails to find a block but the
    manifest insists this file has a flaw -- rather than silently dropping a
    labeled-positive sample, fall back to a context window around the flaw
    line. Logged loudly so it's easy to audit how often this path fires.
    Returns (text, (start_line, end_line)) using the same 1-indexed
    inclusive convention as extract_bad_and_good.
    """
    lines = file_text.splitlines()
    lo = max(0, flaw_line - 1 - window)
    hi = min(len(lines), flaw_line - 1 + window)
    return "\n".join(lines[lo:hi]), (lo + 1, hi)


def _parse_manifest(manifest_path: Path) -> Dict[str, List[Dict]]:
    """Returns {relative_file_path: [{"line": int, "name": str}, ...]}.
    Files with no flaws are present in the dict with an empty list, so
    callers can tell "in manifest, no flaw" apart from "not in manifest"."""
    tree = ET.parse(manifest_path)
    root = tree.getroot()
    file_flaws: Dict[str, List[Dict]] = {}
    for testcase in root.findall("testcase"):
        for file_el in testcase.findall("file"):
            path = file_el.get("path")
            flaws = []
            for flaw_el in file_el.findall("flaw"):
                line = flaw_el.get("line")
                flaws.append({
                    "line": int(line) if line and line.isdigit() else None,
                    "name": flaw_el.get("name"),
                })
            file_flaws.setdefault(path, [])
            file_flaws[path].extend(flaws)
    return file_flaws


def _cwe_id_from_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    m = re.match(r"CWE-(\d+)", name)
    return f"CWE-{m.group(1)}" if m else None


def parse_sard_testsuite(
    root_dir: str,
    manifest_path: str,
    dataset_name: str = "sard",
    default_language: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Sample]:
    """Walk a SARD-format test suite (Juliet or otherwise) and emit one
    Sample per extracted bad() and one per extracted good() function,
    cross-referenced against manifest.xml for CWE id/name and flaw line.

    root_dir: directory the manifest's relative `path` attributes resolve
        against (for the Juliet GitHub mirror, this is the repo root --
        paths look like "testcases/CWE114_.../CWE114_..._01.c").
    """
    root = Path(root_dir)
    manifest = Path(manifest_path)
    file_flaws = _parse_manifest(manifest)

    # Build a filename -> path index ONCE. A naive root.glob("**/<name>")
    # per manifest entry is O(files_in_tree) each time -- fine for one CWE
    # folder, far too slow across the full ~57k-file Juliet tree.
    # Skip .git/ explicitly: walking it wastes time on a git checkout and,
    # in principle, risks a name collision with internal git objects.
    logger.info("Indexing source tree under %s ...", root)
    name_index: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.is_file():
            name_index.setdefault(p.name, p)
    logger.info("Indexed %d files", len(name_index))

    samples: List[Sample] = []
    n_files_seen = 0
    n_fallback = 0
    missing_files: List[str] = []

    for rel_path, flaws in file_flaws.items():
        if limit is not None and n_files_seen >= limit:
            break

        abs_path = name_index.get(Path(rel_path).name)
        if abs_path is None:
            missing_files.append(rel_path)
            continue
        n_files_seen += 1

        text = _read_text(abs_path)
        if text is None:
            continue

        ext = abs_path.suffix.lower()
        language = LANG_BY_EXT.get(ext, default_language or "unknown")

        bad_code, good_code, bad_ranges, good_ranges = extract_bad_and_good(text)
        rel_to_root = str(abs_path.relative_to(root))

        first_flaw = flaws[0] if flaws else None
        cwe_name = first_flaw["name"] if first_flaw else None
        cwe_id = _cwe_id_from_name(cwe_name)
        flaw_line = first_flaw["line"] if first_flaw else None

        if flaws and bad_code is None and flaw_line is not None:
            # Manifest says this file is flawed but our heuristic didn't
            # find an OMITBAD definition block -- fall back rather than
            # silently dropping ground truth, and log it so it's auditable.
            n_fallback += 1
            logger.warning(
                "OMITBAD heuristic miss for %s (flaw at line %s) -- using "
                "line-window fallback", rel_path, flaw_line,
            )
            bad_code, fallback_range = _fallback_line_window(text, flaw_line)
            bad_ranges = [fallback_range]
        elif flaws and bad_code is None:
            # Rarer edge case than the one above: the OMITBAD heuristic ALSO
            # missed, but there's no flaw_line either, so even the fallback
            # window can't be built -- this manifest-labeled-positive file is
            # about to be dropped from the dataset. Every other skip path in
            # this function logs; this one previously didn't. Not confirmed
            # to fire against the real Juliet manifest, but worth being able
            # to see if it ever does.
            logger.warning(
                "No OMITBAD block AND no usable flaw line for %s -- this "
                "manifest-labeled-positive file is being dropped, not "
                "silently included as a false negative.", rel_path,
            )

        if flaws and bad_code:
            samples.append(Sample(
                sample_id=f"{dataset_name}_{abs_path.stem}_bad",
                dataset=dataset_name,
                language=language,
                code=bad_code,
                label=1,
                cwe=cwe_id,
                cwe_name=cwe_name,
                source_file=rel_path,
                flaw_line=flaw_line,
                extra={
                    **({"all_flaws": flaws} if len(flaws) > 1 else {}),
                    "file_path": rel_to_root,
                    "line_ranges": [list(r) for r in bad_ranges],
                },
            ))

        if good_code:
            samples.append(Sample(
                sample_id=f"{dataset_name}_{abs_path.stem}_good",
                dataset=dataset_name,
                language=language,
                code=good_code,
                label=0,
                cwe=cwe_id,        # which CWE this is a *safe* counterexample for
                cwe_name=cwe_name,
                source_file=rel_path,
                flaw_line=None,
                extra={
                    "file_path": rel_to_root,
                    "line_ranges": [list(r) for r in good_ranges],
                },
            ))

    logger.info(
        "Parsed %d/%d manifest files (%d not found on disk, %d used the "
        "line-window fallback) -> %d samples",
        n_files_seen, len(file_flaws), len(missing_files), n_fallback, len(samples),
    )
    if missing_files:
        shown = missing_files[:50]
        suffix = f" (showing first 50 of {len(missing_files)})" if len(missing_files) > 50 else ""
        logger.warning("Files listed in manifest.xml but not found on disk%s:\n%s",
                        suffix, "\n".join(f"  {f}" for f in shown))
    return samples


if __name__ == "__main__":
    print(
        "sard_manifest_parser.py is a library module (the generic "
        "manifest.xml parser) -- it has nothing to run on its own. Use:\n"
        "  python juliet_loader.py --root <juliet_checkout>\n"
        "or call parse_sard_testsuite(...) directly from your own script."
    )
