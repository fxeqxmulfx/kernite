#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.15"
# ///
"""Writes `src/unicode/tables.rs`: what a name is made of and how Rust reads it in NFC, from the Unicode data of the
Python that runs this. Rust 1.100 has Unicode 17.0.0, as Python 3.15 does: the script refuses any other, for the tables
would not be Rust's. `unicode`'s test holds them to what the rustc on the machine says.

The tables hold a name's characters alone, those of XID_Continue from U+0080 up, for no other is in a name. XID_Continue
is closed under NFD, and under composing two of its characters, which the script checks.

    uv run crates/lang/unicode.py"""

import textwrap
import unicodedata
from pathlib import Path

assert unicodedata.unidata_version == "17.0.0", unicodedata.unidata_version
OUT = Path(__file__).resolve().parent / "src/unicode/tables.rs"
HANGUL = range(0xAC00, 0xD7A4)  # the syllables, which the code writes by their rule
CHARS = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
START = {c for c in CHARS if chr(c).isidentifier()}
CONTINUE = {c for c in CHARS if ("a" + chr(c)).isidentifier()}
assert START <= CONTINUE


def ranges(chars: set[int]) -> list[tuple[int, int]]:
    """The runs of consecutive characters from U+0080 up in `chars`, first and last."""
    out: list[list[int]] = []
    for c in sorted(chars):
        if c < 0x80:
            continue
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [(a, b) for a, b in out]


def mapping(c: int) -> list[int]:
    """What `c` is canonically one step, or none."""
    d = unicodedata.decomposition(chr(c))
    return [] if not d or d.startswith("<") else [int(x, 16) for x in d.split()]


def decomposed(c: int) -> list[int]:
    """What `c` is canonically all the way down."""
    return [c] if not mapping(c) else [x for y in mapping(c) for x in decomposed(y)]


def joined(c: int) -> bool:
    """Whether NFC makes `c` of the two characters that it is one step."""
    return len(mapping(c)) == 2 and unicodedata.normalize("NFC", "".join(map(chr, mapping(c)))) == chr(c)


def lit(c: int) -> str:
    return f"'{chr(c)}'" if c < 0x80 and chr(c).isalnum() else f"'\\u{{{c:x}}}'"


def text(cs: list[int]) -> str:
    return '"' + "".join(chr(c) if c < 0x80 and chr(c).isalnum() else f"\\u{{{c:x}}}" for c in cs) + '"'


def table(name: str, about: str, kind: str, rows: list[str]) -> str:
    """A static of `rows`, as many to a line as 120 columns hold."""
    lines, line = [], "   "
    for row in rows:
        if len(line) + len(row) + 2 > 120:
            lines.append(line)
            line = "   "
        line += f" {row},"
    lines.append(line)
    body = "\n".join(lines)
    doc = textwrap.fill(about, 120, initial_indent="/// ", subsequent_indent="/// ")
    return f"{doc}\n#[rustfmt::skip]\npub(super) static {name}: [{kind}; {len(rows)}] = [\n{body}\n];\n"


names = [c for c in CHARS if c in CONTINUE and c >= 0x80]
marks = [(c, unicodedata.combining(chr(c))) for c in names if unicodedata.combining(chr(c))]
steps = [c for c in names if mapping(c) and c not in HANGUL]
pairs = sorted((tuple(mapping(c)), c) for c in steps if joined(c))
assert len({pair for pair, _ in pairs}) == len(pairs)
for c in steps:
    assert all(x in CONTINUE for x in decomposed(c)), hex(c)
for c in filter(joined, CHARS):
    assert c in CONTINUE or not set(mapping(c)) <= CONTINUE, hex(c)
assert all(unicodedata.combining(chr(c)) == 0 for c in HANGUL)

out = """//! The tables of `unicode`, written by `unicode.py` from the Unicode 17.0.0 of Rust 1.100. Do not edit.

use std::ops::RangeInclusive;

"""
out += table(
    "XID_START",
    "The characters from U+0080 up that start a name: XID_Start, in runs.",
    "RangeInclusive<char>",
    [f"{lit(a)}..={lit(b)}" for a, b in ranges(START)],
)
out += "\n" + table(
    "XID_CONTINUE",
    "The characters from U+0080 up that go on in a name: XID_Continue, in runs.",
    "RangeInclusive<char>",
    [f"{lit(a)}..={lit(b)}" for a, b in ranges(CONTINUE)],
)
out += "\n" + table(
    "COMBINING",
    "The characters of a name whose combining class is not 0, with it.",
    "(char, u8)",
    [f"({lit(c)}, {k})" for c, k in marks],
)
out += "\n" + table(
    "DECOMPOSITION",
    "The characters of a name that Unicode says are other ones, canonically, and what they are all the way down, but "
    "the Hangul syllables.",
    "(char, &str)",
    [f"({lit(c)}, {text(decomposed(c))})" for c in steps],
)
out += "\n" + table(
    "COMPOSITION",
    "The pairs of characters of a name that Unicode says make one character, with it: no pair whose character is left "
    "out of NFC, and no Hangul syllable.",
    "((char, char), char)",
    [f"(({lit(a)}, {lit(b)}), {lit(c)})" for (a, b), c in pairs],
)
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(out)
print(f"wrote {OUT}: {len(ranges(START))} + {len(ranges(CONTINUE))} ranges, {len(marks)} marks, ", end="")
print(f"{len(steps)} decompositions, {len(pairs)} compositions")
