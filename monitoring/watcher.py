"""
MIOM Dashboard — monitoring/watcher.py
========================================
Hash-based file watcher for data/live/ and data/processed/.
Uses SHA-256 fingerprints — no watchdog dependency needed for Streamlit.

Called once per rerun; returns True if any watched file has changed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict

from config import LIVE_DIR, PROCESSED_DIR

log = logging.getLogger(__name__)

# File patterns to watch
WATCH_PATTERNS = ["*.csv", "*.txt"]


def _hash_file(path: Path) -> str:
    """SHA-256 of file contents (fast for files up to ~50 MB)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65_536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _scan_dir(directory: Path) -> Dict[str, str]:
    """Returns {relative_path_str: sha256} for all watched files in directory."""
    result: Dict[str, str] = {}
    if not directory.exists():
        return result
    for pattern in WATCH_PATTERNS:
        for p in directory.rglob(pattern):
            key = str(p.relative_to(directory))
            result[key] = _hash_file(p)
    return result


class DataWatcher:
    """
    Lightweight watcher that compares file hashes between calls.
    Call .has_changed() on every Streamlit rerun.
    """

    def __init__(self):
        self._snapshot: Dict[str, str] = {}
        self._watch_dirs = [LIVE_DIR, PROCESSED_DIR]
        self._snapshot = self._current_state()

    def _current_state(self) -> Dict[str, str]:
        state: Dict[str, str] = {}
        for d in self._watch_dirs:
            for rel, digest in _scan_dir(d).items():
                state[f"{d.name}/{rel}"] = digest
        return state

    def has_changed(self) -> bool:
        """
        Returns True (and updates internal snapshot) if any file has
        been added, removed, or modified since the last call.
        """
        current = self._current_state()
        changed = current != self._snapshot

        if changed:
            added   = set(current) - set(self._snapshot)
            removed = set(self._snapshot) - set(current)
            modified = {
                k for k in current
                if k in self._snapshot and current[k] != self._snapshot[k]
            }
            if added:
                log.info("New files detected: %s", added)
            if removed:
                log.info("Files removed: %s", removed)
            if modified:
                log.info("Files modified: %s", modified)
            self._snapshot = current

        return changed

    def watched_file_count(self) -> int:
        return len(self._snapshot)