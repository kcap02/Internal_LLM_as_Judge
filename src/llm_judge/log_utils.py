"""Logging: every stage logs to console AND to logs/<stage>_<timestamp>.log.

The log header records the full run provenance (git commit, config dump,
package versions) so any results file can be traced back to the exact code
and settings that produced it.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone

from .config import LOGS_DIR, REPO_ROOT


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# Distribution names differ from import names for some packages.
_PKGS = {"torch": "torch", "transformers": "transformers",
         "datasets": "datasets", "sklearn": "scikit-learn",
         "numpy": "numpy", "spectral_trust": "spectral_trust"}


def _versions() -> dict:
    """Record package versions WITHOUT importing the packages.

    Importing to read `__version__` has side effects: in the GPU environment
    `import datasets` pulls aiohttp, which crashes on a broken Windows
    certificate store. Provenance logging must never be able to take a run
    down, so read the installed distribution metadata instead.
    """
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name, dist in _PKGS.items():
        try:
            out[name] = version(dist) + _vcs_suffix(dist)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def _vcs_suffix(dist: str) -> str:
    """`@<sha>` for a package installed from a VCS URL, else "".

    C-VER is about provenance that lives only in the log. A version *string*
    is not provenance for a package installed from git: `spectral_trust` reads
    0.2.3 whether it came from the release commit, a later fix, or a local
    edit. pip records the resolved commit in the distribution's
    `direct_url.json` (PEP 610), so read it and pin the record to the actual
    code that ran.
    """
    try:
        from importlib.metadata import distribution
        raw = distribution(dist).read_text("direct_url.json")
        if not raw:
            return ""
        info = json.loads(raw)
        sha = (info.get("vcs_info") or {}).get("commit_id")
        return f"@{sha[:7]}" if sha else ""
    except Exception:
        return ""


def setup_logging(stage: str, config_dump: dict | None = None) -> logging.Logger:
    """Create a logger writing to stdout and logs/<stage>_<UTC timestamp>.log."""
    # Windows consoles default to cp1252 and crash on non-ASCII; force UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logfile = LOGS_DIR / f"{stage}_{stamp}.log"

    logger = logging.getLogger(f"llm_judge.{stage}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(logfile, encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)

    logger.info("stage=%s | commit=%s | logfile=%s", stage, _git_commit(),
                logfile.name)
    logger.info("versions=%s", json.dumps(_versions()))
    if config_dump is not None:
        logger.info("config=%s", json.dumps(config_dump, default=str))
    return logger
