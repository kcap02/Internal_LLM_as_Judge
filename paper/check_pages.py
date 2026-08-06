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
    # No leading \b: a word boundary cannot match between a space and "/", so
    # "under /home/valno" was never flagged. And a single backslash in the
    # regex, not two: r"C:\\\\Users" matches a path containing TWO literal
    # backslashes, which no real path has. Both bugs were silent because a
    # clean document produces no hits either way.
    (r"(?:[A-Za-z]:\\Users\\|/home/|/Users/)[A-Za-z0-9_.\-]+",
     "filesystem path with a username"),
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
    # TWO INDEPENDENT SOURCES, REQUIRED TO AGREE.
    #
    # Detecting the bibliography from rendered text alone failed twice in this
    # project. The ICLR style sets headings in small caps, which pdftotext
    # renders with the letters spaced in ways that shift with the surrounding
    # layout, so a fixed pattern finds nothing -- and the old fallback then
    # counted every page as main text, producing a plausible number from a
    # failed detection. A check whose failure mode is silence is
    # indistinguishable from a check that passed.
    #
    # Primary source is main.aux, where LaTeX records the page of the
    # \label{sec:refstart} placed immediately before \bibliography. The PDF is
    # the cross-check. Neither may be missing, and they must agree.
    aux = PAPER / (Path(args.pdf).stem + ".aux")
    aux_page = None
    if aux.exists():
        m = re.search(r"\\newlabel\{sec:refstart\}\{\{[^}]*\}\{(\d+)\}",
                      aux.read_text(encoding="utf-8", errors="replace"))
        if m:
            aux_page = int(m.group(1))

    def _is_ref_heading(line: str) -> bool:
        return re.sub(r"[^A-Za-z]", "", line).upper() == "REFERENCES"

    pdf_page = next((i for i, p in enumerate(pages, 1)
                     if any(_is_ref_heading(ln) for ln in p.splitlines())),
                    None)

    if aux_page is None:
        print("[FAIL] no \\label{sec:refstart} page in the .aux -- the label is "
              "missing from main.tex, or LaTeX has not been run twice. "
              "REFUSING to guess the main-text length.")
        return 1
    if pdf_page is None:
        print("[FAIL] the References heading was not found in the rendered "
              "text. REFUSING to fall back to counting every page as main "
              "text, which is how this check silently passed twice before.")
        return 1
    # The label sits at the END of the body, so it lands either on the
    # bibliography's first page (no break) or on the page before it (break).
    # Anything else means one of the two readings is wrong.
    if aux_page not in (pdf_page - 1, pdf_page):
        print(f"[FAIL] sources disagree beyond a page break: the "
              f"\\label{{sec:refstart}} is on page {aux_page}, the rendered "
              f"References heading is on page {pdf_page}")
        return 1

    ref_page = pdf_page
    main_pages = ref_page - 1
    status = "PASS" if main_pages <= args.limit else "FAIL"
    print(f"[{status}] main text {main_pages} pages (limit {args.limit}); "
          f"bibliography starts on page {ref_page} (.aux and PDF agree)")
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
