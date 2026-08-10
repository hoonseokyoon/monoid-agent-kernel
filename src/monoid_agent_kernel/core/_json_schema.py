"""The one regex assertion that means the same thing in both dialects a ``pattern`` is read in.

JSON Schema calls ``pattern`` an ECMA-262 regular expression and applies it unanchored, but says
nothing about which engine runs it -- and the two that matter disagree at exactly one position.
Python's ``re``, which ``jsonschema`` uses via ``re.search``, lets ``$`` match immediately before a
single trailing newline; ECMA-262's ``$`` without the ``m`` flag matches only at end of input. So
``^(|body)$`` certifies ``"value\\n"`` under one engine and refuses it under the other, and a
record every other edge in this kernel rejects walks through ``monoid validate``.

``\\Z`` is the Python spelling of "end of input, and nothing may follow", and it is exactly the
wrong choice here: ECMA-262 has no ``\\Z``, it is an identity escape there, so a published schema
ending in ``\\Z`` would demand a literal ``Z`` from any JavaScript validator reading it. This
suffix is what both engines agree on -- ``$`` for the reader, then an assertion that no character
of any kind follows, which is redundant under ECMA-262 and load-bearing under ``re``.

Deliberately a shared constant rather than four spellings of it. A trailing-newline hole is
invisible at the site that has it: the pattern reads correct, the field validates, and only a
value nobody writes by hand reveals the difference. Stated once, it can be fixed once.
"""

from __future__ import annotations

END_OF_INPUT = r"$(?![\s\S])"
