"""Result persistence: append-only JSONL streams + atomic JSON for artefacts.

Two different jobs, two different mechanisms.

**Result streams (JSONL).** GPU stages emit one row per item and must survive
a Ctrl-C or an OOM at any moment. Rewriting a growing JSON array after every
item is O(n^2) and, on Windows, loses rows outright: the rename step of the
write-tmp-then-replace dance intermittently fails with `PermissionError
[WinError 5]` because an antivirus or the search indexer still holds the
freshly written file. Appending a single line per row is O(1), never
rewrites, and cannot be interrupted mid-array.

**Artefacts (JSON).** Banks, analyses and audits are written once, whole, and
are meant to be read by humans — those stay pretty-printed JSON, written
through a retrying atomic replace.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def atomic_write_json(path: str | Path, obj, retries: int = 8) -> None:
    """Write JSON via tmp+replace, retrying the replace on transient locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    delay = 0.05
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_rows(path: str | Path) -> list[dict]:
    """Read a result stream: `<path>.jsonl` plus any legacy `<path>.json`.

    Both are read and concatenated so runs started before the JSONL switch
    are not orphaned.
    """
    path = Path(path)
    base = path.with_suffix("")
    rows: list[dict] = []

    legacy = base.with_suffix(".json")
    if legacy.exists():
        try:
            data = read_json(legacy) or []
            rows.extend(data if isinstance(data, list) else [])
        except json.JSONDecodeError:
            pass  # truncated legacy file: ignore rather than abort the run

    stream = base.with_suffix(".jsonl")
    if stream.exists():
        with open(stream, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A partially flushed final line after a hard kill:
                    # drop it, keep everything before it.
                    continue
    return rows


class ResumableResults:
    """Append-only result stream with (model, item_id) resume keys."""

    def __init__(self, path: str | Path, key_fields=("model", "item_id")):
        base = Path(path).with_suffix("")
        self.path = base.with_suffix(".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_fields = key_fields
        self.rows = read_rows(self.path)
        self.done = {tuple(r.get(k) for k in key_fields) for r in self.rows}
        self._fh = None

    def _handle(self):
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        return self._fh

    def is_done(self, *key) -> bool:
        return tuple(key) in self.done

    def append(self, row: dict) -> None:
        fh = self._handle()
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()          # survive a kill; the OS buffer is not enough
        self.rows.append(row)
        self.done.add(tuple(row.get(k) for k in self.key_fields))

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __len__(self) -> int:
        return len(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
