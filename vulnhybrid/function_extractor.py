"""
Turn an arbitrary C/C++ source file into function-level units.

The thesis pipeline evaluates at *function* granularity -- Juliet's ``bad()``
/ ``good()`` bodies are sliced out via ``#ifndef OMITBAD`` guards, Devign only
ever shipped extracted functions. A real scanner has neither of those
crutches: it gets a messy ``.c``/``.cpp`` file and has to find the functions
itself. That is the one genuinely new capability the practical tool needs, and
it lives here.

Two backends, chosen at runtime:

  * **tree-sitter** (preferred): a real incremental parser with proper C and
    C++ grammars. Robust against the things that defeat regexes -- function
    pointers in parameter lists, C++ member/operator/destructor definitions,
    braces inside strings and comments, macro noise. Used automatically when
    ``tree_sitter`` + ``tree_sitter_c`` / ``tree_sitter_cpp`` are importable.
  * **brace matcher** (zero-dependency fallback): a masking scanner that
    blanks out comments, string/char literals and preprocessor lines, then
    pairs braces to recover function bodies. Handles the common cases; it is
    deliberately conservative and is documented as approximate. This mirrors
    the pipeline's own philosophy of always having a local fallback for an
    optional dependency (HuggingFace->local jsonl, Semgrep registry->local
    rules).

If a static finding lands *outside* any function we can recover (file-scope
code, a macro body, a parser miss), the caller can fall back to
``window_unit`` -- a fixed window of lines around the flagged line -- which is
exactly the ``flaw_line +/- window`` strategy the thesis uses when Juliet's
guard-based extraction fails (Section 3-2-1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

# Extension -> language string used both to pick a grammar and to label the
# code block in the LLM prompt (the prompt fences the snippet as ```{language}).
_C_EXTS = {".c"}
_CPP_EXTS = {".cpp", ".cc", ".cxx", ".c++", ".cp", ".hpp", ".hh", ".hxx",
             ".h++", ".tcc", ".ipp", ".inl"}
# .h is ambiguous. Default it to C++: the C++ grammar is a near-superset and
# parses plain C headers fine, whereas the C grammar chokes on C++-only
# constructs. The prompt still says "c" for .h (see detect_language) because
# most .h in this project's corpus are C.
_HEADER_AMBIG = {".h"}

_ALL_SOURCE_EXTS = _C_EXTS | _CPP_EXTS | _HEADER_AMBIG


def is_source_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _ALL_SOURCE_EXTS


def detect_language(path: str) -> str:
    """Return ``"c"`` or ``"cpp"`` for the LLM prompt / grammar choice."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _CPP_EXTS:
        return "cpp"
    return "c"  # .c and ambiguous .h both default to C for labeling


def _grammar_language(path: str) -> str:
    """Which tree-sitter grammar to parse with. Ambiguous .h uses the C++
    grammar (superset) even though we *label* it C."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _CPP_EXTS or ext in _HEADER_AMBIG:
        return "cpp"
    return "c"


# ---------------------------------------------------------------------------
# The unit of analysis
# ---------------------------------------------------------------------------


@dataclass
class Function:
    """One extracted unit. ``start_line``/``end_line`` are 1-indexed and
    inclusive, referring to line numbers in the *original file* -- so they line
    up directly with the file:line a static analyzer reports."""

    name: str
    code: str
    start_line: int
    end_line: int
    file: str
    language: str
    kind: str = "function"        # "function" | "window"
    extra: Dict[str, object] = field(default_factory=dict)

    def contains(self, file_line: int) -> bool:
        return self.start_line <= file_line <= self.end_line

    def snippet_line(self, file_line: int) -> Optional[int]:
        """Map a 1-indexed line in the original file to a 1-indexed line
        within this unit's own ``code`` text (what a report should show), or
        ``None`` if the line is outside this unit."""
        if self.contains(file_line):
            return file_line - self.start_line + 1
        return None

    @property
    def span(self) -> int:
        return self.end_line - self.start_line + 1


# ---------------------------------------------------------------------------
# tree-sitter backend
# ---------------------------------------------------------------------------

_TS_PARSERS: Dict[str, object] = {}
_TS_AVAILABLE: Optional[bool] = None


def tree_sitter_available() -> bool:
    """True if the tree-sitter backend can be used. Cached after first probe."""
    global _TS_AVAILABLE
    if _TS_AVAILABLE is None:
        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_c  # noqa: F401
            import tree_sitter_cpp  # noqa: F401
            _TS_AVAILABLE = True
        except Exception:
            _TS_AVAILABLE = False
    return _TS_AVAILABLE


def _get_parser(grammar: str):
    if grammar in _TS_PARSERS:
        return _TS_PARSERS[grammar]
    from tree_sitter import Parser, Language
    if grammar == "cpp":
        import tree_sitter_cpp as ts_lang
    else:
        import tree_sitter_c as ts_lang
    parser = Parser(Language(ts_lang.language()))
    _TS_PARSERS[grammar] = parser
    return parser


_NAME_NODE_TYPES = {
    "identifier", "field_identifier", "type_identifier",
    "qualified_identifier", "destructor_name", "operator_name",
    "operator_cast",
}


def _ts_function_name(node) -> str:
    """Best-effort function name from a ``function_definition`` node, following
    the declarator chain down to the identifier (handles ``*foo``, ``Foo::bar``,
    ``~Foo``, ``operator==``)."""
    declarator = node.child_by_field_name("declarator")

    def find_name(n):
        if n is None:
            return None
        if n.type in _NAME_NODE_TYPES:
            return n.text.decode("utf-8", "replace")
        child = n.child_by_field_name("declarator")
        if child is not None:
            r = find_name(child)
            if r:
                return r
        for c in n.children:
            r = find_name(c)
            if r:
                return r
        return None

    return find_name(declarator) or "<anonymous>"


def _ts_extract(source: bytes, grammar: str, path: str, language: str) -> List[Function]:
    parser = _get_parser(grammar)
    tree = parser.parse(source)
    functions: List[Function] = []

    # Iterative pre-order walk. We record every function_definition but do NOT
    # descend into one once found -- a lambda or local class inside a function
    # belongs to that function, which is the unit we want to hand the model.
    def visit(node):
        if node.type == "function_definition":
            code = source[node.start_byte:node.end_byte].decode("utf-8", "replace")
            functions.append(Function(
                name=_ts_function_name(node),
                code=code,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                file=path,
                language=language,
            ))
            return
        for c in node.children:
            visit(c)

    visit(tree.root_node)
    functions.sort(key=lambda f: (f.start_line, f.end_line))
    return functions


# ---------------------------------------------------------------------------
# Zero-dependency brace-matching backend
# ---------------------------------------------------------------------------

# Identifiers that can appear as ``name(`` but never start a function
# *definition* -- control flow, operators, and common macros that look callable.
_NON_FUNCTION_IDENTS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "alignof",
    "decltype", "typeof", "typeof_unqual", "static_assert", "_Static_assert",
    "and", "or", "not", "xor", "bitand", "bitor", "compl",
    "__attribute__", "__declspec", "__asm__", "asm", "throw", "new", "delete",
    "case", "goto", "do", "else",
}


def _mask(src: str) -> str:
    """Return a copy of ``src`` with comment bodies, string/char literal
    contents and preprocessor directives replaced by spaces (newlines kept, so
    character offsets and line numbers are preserved). Braces and parens inside
    those regions then can't corrupt the matching."""
    out = list(src)
    n = len(src)
    i = 0
    state = None  # None | 'line' | 'block' | 'str' | 'chr' | 'pp'

    def at_line_start(idx: int) -> bool:
        j = idx - 1
        while j >= 0 and src[j] != "\n":
            if not src[j].isspace():
                return False
            j -= 1
        return True

    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state is None:
            if c == "/" and nxt == "/":
                out[i] = out[i + 1] = " "; i += 2; state = "line"; continue
            if c == "/" and nxt == "*":
                out[i] = out[i + 1] = " "; i += 2; state = "block"; continue
            if c == '"':
                i += 1; state = "str"; continue          # keep the opening quote
            if c == "'":
                i += 1; state = "chr"; continue
            if c == "#" and at_line_start(i):
                out[i] = " "; i += 1; state = "pp"; continue
            i += 1; continue

        if state == "line":
            if c == "\n":
                state = None
            else:
                out[i] = " "
            i += 1; continue

        if state == "block":
            if c == "*" and nxt == "/":
                out[i] = out[i + 1] = " "; i += 2; state = None; continue
            if c != "\n":
                out[i] = " "
            i += 1; continue

        if state in ("str", "chr"):
            closer = '"' if state == "str" else "'"
            if c == "\\":
                out[i] = " "
                if i + 1 < n and src[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2; continue
            if c == closer:
                i += 1; state = None; continue           # keep the closing quote
            if c != "\n":
                out[i] = " "
            i += 1; continue

        if state == "pp":
            if c == "\\" and nxt == "\n":
                out[i] = " "; i += 2; continue            # line continuation
            if c == "\n":
                state = None; i += 1; continue            # keep the newline
            out[i] = " "; i += 1; continue

    return "".join(out)


def _match(masked: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index of the ``close_ch`` matching the ``open_ch`` at ``start``, or -1."""
    depth = 0
    n = len(masked)
    i = start
    while i < n:
        ch = masked[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _line_index(src: str) -> List[int]:
    """char offset -> 1-indexed line number, as a list indexed by offset."""
    line_of = [1] * (len(src) + 1)
    line = 1
    for idx, ch in enumerate(src):
        line_of[idx] = line
        if ch == "\n":
            line += 1
    line_of[len(src)] = line
    return line_of


def _stmt_start(masked: str, name_start: int) -> int:
    """Walk backward from the start of the function name to the beginning of
    its signature -- i.e. just after the previous ``;``, ``}``, ``{`` or the
    start of file -- so the recorded code includes the return type."""
    k = name_start - 1
    while k >= 0 and masked[k] not in ";{}":
        k -= 1
    start = k + 1
    while start < name_start and masked[start] in " \t\r\n":
        start += 1
    return start


def _qualified_name(src: str, name_start: int, name_end: int) -> str:
    """Grow a bare identifier leftward across ``::`` and a leading ``~`` so a
    method reads as ``Class::method`` and a destructor as ``~Class``."""
    start = name_start
    # absorb  Foo:: chains
    j = name_start - 1
    while j >= 1 and src[j] == ":" and src[j - 1] == ":":
        k = j - 2
        while k >= 0 and (src[k].isalnum() or src[k] == "_"):
            k -= 1
        start = k + 1
        j = k
        while j >= 0 and src[j] in " \t":
            j -= 1
    if start - 1 >= 0 and src[start - 1] == "~":
        start -= 1
    return src[start:name_end].strip()


def _brace_extract(source: str, path: str, language: str) -> List[Function]:
    masked = _mask(source)
    n = len(masked)
    line_of = _line_index(source)
    functions: List[Function] = []
    i = 0
    while i < n:
        if masked[i] != "(":
            i += 1
            continue
        close_paren = _match(masked, i, "(", ")")
        if close_paren == -1:
            break

        # Identifier immediately before '(' ?
        k = i - 1
        while k >= 0 and masked[k] in " \t\r\n":
            k -= 1
        name_end = k + 1
        while k >= 0 and (masked[k].isalnum() or masked[k] == "_"):
            k -= 1
        name_start = k + 1
        name = source[name_start:name_end]

        if (not name or not (name[0].isalpha() or name[0] == "_")
                or name in _NON_FUNCTION_IDENTS):
            i = close_paren + 1
            continue

        # After ')', the next significant delimiter decides: '{' => definition,
        # ';' => prototype/declaration. Everything between (qualifiers, a C++
        # ctor init list, trailing return type) is skipped over.
        p = close_paren + 1
        while p < n and masked[p] not in ";{":
            p += 1
        if p < n and masked[p] == "{":
            body_end = _match(masked, p, "{", "}")
            if body_end == -1:
                i = close_paren + 1
                continue
            sig_start = _stmt_start(masked, name_start)
            functions.append(Function(
                name=_qualified_name(source, name_start, name_end),
                code=source[sig_start:body_end + 1],
                start_line=line_of[sig_start],
                end_line=line_of[body_end],
                file=path,
                language=language,
            ))
            i = body_end + 1
        else:
            i = (p + 1) if p < n else close_paren + 1

    functions.sort(key=lambda f: (f.start_line, f.end_line))
    return functions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_functions(source: str, path: str = "<memory>",
                      language: Optional[str] = None,
                      prefer_tree_sitter: bool = True) -> List[Function]:
    """Extract every function definition from ``source``.

    Uses tree-sitter when available (and ``prefer_tree_sitter``), otherwise the
    brace-matching fallback. ``language`` overrides extension-based detection.
    """
    lang = language or detect_language(path)
    if prefer_tree_sitter and tree_sitter_available():
        try:
            grammar = _grammar_language(path) if language is None else (
                "cpp" if lang == "cpp" else "c")
            return _ts_extract(source.encode("utf-8", "replace"), grammar, path, lang)
        except Exception:
            # Never let a parser hiccup take down a scan -- degrade to the
            # dependency-free path, same spirit as the pipeline's fallbacks.
            pass
    return _brace_extract(source, path, lang)


def extract_functions_from_file(path: str,
                                prefer_tree_sitter: bool = True) -> List[Function]:
    with open(path, encoding="utf-8", errors="replace") as f:
        source = f.read()
    return extract_functions(source, path=path, prefer_tree_sitter=prefer_tree_sitter)


def enclosing_function(functions: List[Function], file_line: int) -> Optional[Function]:
    """Smallest-span function containing ``file_line`` (innermost, for nested
    definitions), or ``None`` if the line is outside every function."""
    best: Optional[Function] = None
    for fn in functions:
        if fn.contains(file_line):
            if best is None or fn.span < best.span:
                best = fn
    return best


def window_unit(source: str, path: str, file_line: int, language: Optional[str] = None,
                radius: int = 8) -> Function:
    """Fallback unit for a finding that isn't inside any recovered function: a
    +/- ``radius``-line window around ``file_line``. Mirrors the thesis's
    ``flaw_line +/- window`` fallback when guard-based extraction fails."""
    lang = language or detect_language(path)
    lines = source.splitlines()
    total = len(lines)
    start = max(1, file_line - radius)
    end = min(total, file_line + radius) if total else file_line
    code = "\n".join(lines[start - 1:end])
    return Function(
        name=f"{os.path.basename(path)}:{file_line} (context window)",
        code=code,
        start_line=start,
        end_line=end,
        file=path,
        language=lang,
        kind="window",
    )
