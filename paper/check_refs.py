"""Fail the build on any citation that has not been verified to exist.

Same discipline as check_macros.py, applied to the bibliography. A fabricated
or misremembered citation is a silent correctness failure producing entirely
plausible output — the exact failure mode this paper is about — and it is the
one that ends a submission rather than costing a revision.

Rules:
  * every \\cite{} / \\citep{} / \\citet{} key in main.tex must exist in refs.bib;
  * every entry in refs.bib must carry a `doi` or `eprint` (arXiv) field;
  * a key that cannot be resolved to a real paper is deleted, together with the
    sentence citing it. A claim that existed only to be supported by an
    unverifiable citation was not a claim worth keeping.

Run:  python paper/check_refs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent


def main() -> int:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib_path = PAPER / "refs.bib"
    bib = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""

    cited: set[str] = set()
    for keys in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        cited.update(k.strip() for k in keys.split(",") if k.strip())

    entries = dict(re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S))
    verified = {k.strip() for k, body in entries.items()
                if re.search(r"^\s*(doi|eprint)\s*=", body, re.M | re.I)}
    unverified_entries = sorted(set(entries) - verified)

    missing = sorted(cited - set(entries))
    present_unverified = sorted((cited & set(entries)) - verified)

    print(f"{len(cited)} keys cited, {len(entries)} entries in refs.bib, "
          f"{len(verified)} carrying a DOI or arXiv ID")

    if missing:
        print(f"\nNOT IN refs.bib ({len(missing)}) — these have not been "
              f"verified to be real papers:")
        for k in missing:
            print(f"  {k}")
    if present_unverified:
        print(f"\nNO IDENTIFIER ({len(present_unverified)}) — present but "
              f"without a DOI or arXiv ID:")
        for k in present_unverified:
            print(f"  {k}")
    if unverified_entries:
        print(f"\n(entries lacking an identifier, cited or not: "
              f"{len(unverified_entries)})")

    if missing or present_unverified:
        print("\nBUILD BLOCKED. Verify each key against the actual paper, or "
              "delete it and the sentence citing it.")
        return 1
    print("\nEvery citation resolves to a verified entry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
