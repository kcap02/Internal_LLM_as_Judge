"""Result persistence: atomic JSON writes + resume support.

A Ctrl-C during json.dump would truncate the file and lose the whole run, not
just the last item — hence write-to-tmp-then-os.replace. Every GPU stage is
resumable: on restart, (model, item_id) pairs already present are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class ResumableResults:
    """Append-only result store with (model, item_id) resume keys."""

    def __init__(self, path: str | Path, key_fields=("model", "item_id")):
        self.path = Path(path)
        self.key_fields = key_fields
        self.rows: list[dict] = read_json(self.path, default=[]) or []
        self.done = {tuple(r[k] for k in key_fields) for r in self.rows}

    def is_done(self, *key) -> bool:
        return tuple(key) in self.done

    def append(self, row: dict) -> None:
        self.rows.append(row)
        self.done.add(tuple(row[k] for k in self.key_fields))
        atomic_write_json(self.path, self.rows)

    def __len__(self) -> int:
        return len(self.rows)
