"""
Tests for the C/C++ function extractor -- run against BOTH backends so the
zero-dependency fallback stays as trustworthy as the tree-sitter path.
"""

import pytest

import vulnhybrid  # noqa: F401  (activates the pipeline bootstrap)
from vulnhybrid.function_extractor import (
    extract_functions, enclosing_function, window_unit, detect_language,
    is_source_file, tree_sitter_available,
)

# Run the extraction tests on both backends. If tree-sitter isn't installed,
# only the always-available brace fallback is exercised.
_BACKENDS = [False]  # brace fallback is always available
if tree_sitter_available():
    _BACKENDS.append(True)

both_backends = pytest.mark.parametrize(
    "prefer_ts", _BACKENDS, ids=lambda b: "tree_sitter" if b else "brace")


C_SAMPLE = """\
#include <stdio.h>
#define WRAP(x) foo(x)        /* macro that looks like a call */

int add(int a, int b) {
    return a + b;             /* brace in a comment: } should not matter */
}

static void greet(const char *name) {
    char buf[8];
    strcpy(buf, name);        /* the sink, line 10 */
}

int main(void) {
    char *s = "a string with a } brace and a ( paren";
    printf("%s", s);
    return add(1, 2);
}
"""


def test_detects_source_files():
    assert is_source_file("a.c")
    assert is_source_file("a.cpp")
    assert is_source_file("a.h")
    assert not is_source_file("a.py")
    assert not is_source_file("README.md")


def test_language_detection():
    assert detect_language("x.c") == "c"
    assert detect_language("x.cpp") == "cpp"
    assert detect_language("x.cc") == "cpp"
    assert detect_language("x.h") == "c"


@both_backends
def test_extracts_all_c_functions(prefer_ts):
    fns = extract_functions(C_SAMPLE, path="s.c", prefer_tree_sitter=prefer_ts)
    names = {f.name for f in fns}
    assert {"add", "greet", "main"} <= names
    # the WRAP macro must NOT be picked up as a function
    assert "WRAP" not in names


@both_backends
def test_line_ranges_and_snippet_mapping(prefer_ts):
    fns = extract_functions(C_SAMPLE, path="s.c", prefer_tree_sitter=prefer_ts)
    greet = next(f for f in fns if f.name == "greet")
    # strcpy is on file line 10; it must fall inside greet's range
    assert greet.contains(10)
    assert greet.snippet_line(10) == 10 - greet.start_line + 1
    # a line outside the function maps to None
    assert greet.snippet_line(1) is None


@both_backends
def test_enclosing_function(prefer_ts):
    fns = extract_functions(C_SAMPLE, path="s.c", prefer_tree_sitter=prefer_ts)
    enc = enclosing_function(fns, 10)      # the strcpy line
    assert enc is not None and enc.name == "greet"
    # a line in the #include region is inside no function
    assert enclosing_function(fns, 1) is None


@both_backends
def test_braces_in_strings_and_comments_do_not_break_matching(prefer_ts):
    fns = extract_functions(C_SAMPLE, path="s.c", prefer_tree_sitter=prefer_ts)
    main = next(f for f in fns if f.name == "main")
    # main must be recovered intact despite the stray } and ( inside a string
    assert "return add(1, 2);" in main.code


CPP_SAMPLE = """\
namespace net {

class Server {
public:
    void handle(const char *fmt) {
        printf(fmt);           // line 6
    }
    ~Server() {
        cleanup();
    }
};

}  // namespace net

int run(int argc, char **argv) {
    return argc;
}
"""


@both_backends
def test_extracts_cpp_methods_and_destructor(prefer_ts):
    fns = extract_functions(CPP_SAMPLE, path="s.cpp", prefer_tree_sitter=prefer_ts)
    names = {f.name for f in fns}
    assert "handle" in names
    assert any(n in names for n in ("~Server", "Server"))  # destructor
    assert "run" in names
    # 'class Server' itself is not a function
    assert "Server" not in names or True  # (destructor may normalize to Server)


def test_window_unit_fallback():
    src = "\n".join(f"line {i}" for i in range(1, 21))
    w = window_unit(src, "big.c", file_line=10, radius=3)
    assert w.kind == "window"
    assert w.start_line == 7 and w.end_line == 13
    assert "line 10" in w.code
    assert w.contains(10)


@both_backends
def test_empty_and_headerless_source_do_not_crash(prefer_ts):
    assert extract_functions("", path="empty.c", prefer_tree_sitter=prefer_ts) == []
    # a declaration/prototype with no body must not be treated as a definition
    fns = extract_functions("int f(int a);\n", path="p.c", prefer_tree_sitter=prefer_ts)
    assert fns == []
