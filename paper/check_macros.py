"""Fail the build if main.tex prints a macro that is undefined or unmeasured.

Two distinct failures, both of which silently produce a plausible-looking PDF:
  * a macro used in the text but never defined -> LaTeX prints nothing at all;
  * a macro defined but still PENDING -> the PDF prints "??" where a measured
    quantity should be, and a reader cannot tell it from a typographic glitch.

Run before every build:  python paper/check_macros.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent


def main() -> int:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    gen_path = PAPER / "generated_numbers.tex"
    if not gen_path.exists():
        print("generated_numbers.tex missing — run paper/fill_numbers.py")
        return 1
    gen = gen_path.read_text(encoding="utf-8")

    defs = dict(re.findall(r"newcommand\{\\([A-Za-z]+)\}\{(.*)\}", gen))
    # Body macros are written \Foo{} by convention, which is also what makes
    # them greppable; LaTeX builtins are excluded by that trailing {}.
    used = set(re.findall(r"\\([A-Z][A-Za-z]*)\{\}", tex))

    undefined = sorted(used - set(defs))
    pending = sorted(m for m in used if "??" in defs.get(m, ""))

    print(f"{len(defs)} macros defined, {len(used)} used in main.tex")
    if undefined:
        print(f"\nUNDEFINED ({len(undefined)}) — used in the text, never defined:")
        for m in undefined:
            print(f"  \\{m}")
    if pending:
        print(f"\nPENDING ({len(pending)}) — defined but not yet measured; "
              f"these print as ?? in the PDF:")
        for m in pending:
            print(f"  \\{m}")

    unused = sorted(set(defs) - used - {m for m in defs if m in used})
    if unused:
        print(f"\n(unused definitions: {len(unused)} — harmless)")

    if undefined or pending:
        print("\nBUILD BLOCKED.")
        return 1
    print("\nAll macros defined and measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
