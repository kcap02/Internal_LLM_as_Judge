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
from datetime import datetime
from pathlib import Path

from .config import LOGS_DIR, REPO_ROOT


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "datasets", "sklearn", "numpy",
                "spectral_trust"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except ImportError:
            out[mod] = "not installed"
    return out


def setup_logging(stage: str, config_dump: dict | None = None) -> logging.Logger:
    """Create a logger writing to stdout and logs/<stage>_<UTC timestamp>.log."""
    # Windows consoles default to cp1252 and crash on non-ASCII; force UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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
