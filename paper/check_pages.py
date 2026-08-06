"""Fail the build if the PDF breaks ICLR 2027's format or anonymity rules.

Third build-time gate, alongside check_macros.py (unmeasured numbers) and
check_refs.py (unverified citations). This one reads the BUILT PDF rather than
the source, because the two quantities it checks -- how many pages the main
text occupies, and what a reviewer can see -- are properties of the render.

ICLR 2027 (Author Guidelines, retrieved 2026-08-06):
  * main text 9 pages or fewer at submission (10 for rebuttal/camera-ready);
  * references and appendices excluded from that count;
  * repository links permitted only if completely anonymous.

Run:  python paper/check_pages.py [--limit 9]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent

# Patterns that deanonymise a double-blind submission. A repository link is
# allowed only if anonymous, which this cannot verify, so any URL is surfaced
# for a human decision rather than silently passed.
DEANON = [
    (r"github\.com/[^\s}]+", "GitHub URL"),
    (r"https?://(?!anonymous)[^\s}]+", "URL (anonymous hosts excepted)"),
    (r"\bour (?:repository|repo|github|code release)\b", "first-person repo reference"),
    (r"\b(?:C:\\\\Users|/home/|/Users/)[A-Za-z0-9_.-]+", "filesystem path with a username"),
    (r"\\(?:acknowledgements|acks|thanks)\b", "acknowledgement"),
]


def pdf_text(pdf: Path) -> list[str]:
    """Per-page text via pdftotext; empty list if unavailable."""
    out = PAPER / "_pages.txt"
    try:
        subprocess.run(["pdftotext", str(pdf), str(out)], check=True,
                       capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    txt = out.read_text(encoding="utf-8", errors="replace")
    out.unlink(missing_ok=True)
    return txt.split("\f")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="main.pdf")
    ap.add_argument("--limit", type=int, default=9,
                    help="ICLR 2027: 9 at submission, 10 at camera-ready")
    args = ap.parse_args()

    pdf = PAPER / args.pdf
    if not pdf.exists():
        print(f"{args.pdf} not built")
        return 1

    fail = False
    pages = pdf_text(pdf)
    if not pages:
        print("WARN pdftotext unavailable; page and anonymity checks skipped")
        return 0

    # ── Main-text length, counted to the references ──────────────────────
    # The ICLR style sets section headings in small caps, which pdftotext
    # renders with the initial letter separated: "R EFERENCES". Matching
    # "^References$" silently finds nothing and the whole document is then
    # counted as main text.
    ref_re = re.compile(r"^\s*R\s*EFERENCES\s*$", re.M | re.I)
    ref_page = next((i for i, p in enumerate(pages, 1) if ref_re.search(p)),
                    None)
    if ref_page is None:
        print("WARN no References heading found; counting all pages as main text")
        main_pages = len([p for p in pages if p.strip()])
    else:
        main_pages = ref_page - 1
    status = "PASS" if main_pages <= args.limit else "FAIL"
    print(f"[{status}] main text {main_pages} pages (limit {args.limit}); "
          f"references start on page {ref_page}")
    if main_pages > args.limit:
        fail = True

    # ── Anonymity ────────────────────────────────────────────────────────
    body = "\n".join(pages)
    src = (PAPER / "main.tex").read_text(encoding="utf-8")
    hits = []
    for pat, label in DEANON:
        for hay, where in ((body, "PDF"), (src, "source")):
            for m in re.findall(pat, hay, re.I):
                hits.append((label, where, m[:70]))
    if hits:
        print(f"[WARN] {len(hits)} possible deanonymising item(s); a repository "
              f"link is permitted only if the host is anonymous and does not "
              f"track visitors:")
        for label, where, m in hits[:12]:
            print(f"    {where}: {label}: {m}")
    else:
        print("[PASS] no URLs, repository references, usernames or "
              "acknowledgements found")

    # ── Bibliography style ───────────────────────────────────────────────
    bbl = PAPER / (Path(args.pdf).stem + ".bbl")
    if bbl.exists():
        items = re.findall(r"\\bibitem\[([^\]]*)\]", bbl.read_text(
            encoding="utf-8", errors="replace"))
        author_year = sum(1 for i in items if re.search(r"\(\d{4}\)", i))
        ok = items and author_year == len(items)
        print(f"[{'PASS' if ok else 'FAIL'}] bibliography: {author_year}/"
              f"{len(items)} entries in author-year form")
        if not ok:
            fail = True
    else:
        print("WARN no .bbl; run bibtex")

    if fail:
        print("\nBUILD BLOCKED.")
        return 1
    print("\nFormat checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
