"""Does every anonymity pattern fire on planted content?"""
import importlib.util
import re
import sys

spec = importlib.util.spec_from_file_location(
    "cp", r"c:\Users\valno\Dev\Internal_LLM_as_Judge\paper\check_pages.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)

PLANTED = [
    ("github url", "Code at https://github.com/example/judge-internals."),
    ("bare url", "See https://example.org/data for the archive."),
    ("first-person repo", "Our repository contains the full pipeline."),
    ("unix path", r"data under /home/valno/Dev/data"),
    ("mac path", r"see /Users/valno/Dev/out"),
    ("windows path", r"see C:\Users\valno\Dev\out"),
    ("acknowledgement", r"\acks{We thank our funders.}"),
    ("thanks macro", r"\thanks{Supported by a grant.}"),
]

print(f"{'planted item':<20} {'fires?':<8} matched")
print("-" * 62)
missed = []
for label, text in PLANTED:
    hits = []
    for pat, plabel in cp.DEANON:
        hits += re.findall(pat, text, re.I)
    ok = bool(hits)
    if not ok:
        missed.append(label)
    print(f"{label:<20} {'YES' if ok else 'NO':<8} {hits[:1]}")

print()
if missed:
    print("SILENT PATTERNS:", ", ".join(missed))
    sys.exit(1)
print("every planted item fires")
