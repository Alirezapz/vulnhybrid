"""Label-blind C/C++ preprocessing for research, never a compiler rewrite.

Remove comments without shifting lines. Keep literals, operators and ordinary
identifiers. Juliet benchmark identifiers are optionally replaced consistently
inside each snippet. Raw C++ strings and comment markers inside strings survive.
This mitigates explicit hints; it cannot rule out pretraining contamination.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

_LEX = re.compile(
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delim>[^\s()\\]{0,16})\(.*?\)(?P=delim)")'
    r'|(?P<string>(?:u8|u|U|L)?"(?:\\.|[^"\\])*")'
    r"|(?P<char>(?:u|U|L)?'(?:\\.|[^'\\])*')"
    r'|(?P<comment>//[^\n]*|/\*.*?\*/)'
    r'|(?P<identifier>[A-Za-z_][A-Za-z_0-9]*)'
    r'|(?P<number>(?:0[xX][0-9a-fA-F]+|(?:\d+\.\d*|\.\d+|\d+)(?:[eEpP][+-]?\d+)?)[uUlLfF]*)'
    r'|(?P<operator>>>=|<<=|->\*|\.\.\.|::|\+\+|--|->|&&|\|\||==|!=|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|##)'
    r'|(?P<other>.)', re.DOTALL,
)


def sanitize_code(code: str, juliet: bool = False) -> str:
    names = {}
    parts = []
    for m in _LEX.finditer(code):
        token = m.group()
        if m.group('comment') is not None:
            token = ''.join('\n' if c == '\n' else ' ' for c in token)
        elif juliet and m.group('identifier') is not None:
            if (re.search(r'CWE\d+|(?:^|_)(?:bad|good)', token, re.I)
                    or token in ('OMITBAD', 'OMITGOOD')):
                if token not in names:
                    names[token] = f'benchmark_symbol_{len(names)}'
                token = names[token]
        parts.append(token)
    return ''.join(parts)


@lru_cache(maxsize=40000)
def code_hash(code: str) -> str:
    # Token boundaries are preserved: `a b` cannot collide with `ab`, and
    # whitespace inside string literals remains significant.
    clean = sanitize_code(code)
    tokens = [m.group() for m in _LEX.finditer(clean) if not m.group().isspace()]
    return hashlib.sha256('\x1f'.join(tokens).encode()).hexdigest()
