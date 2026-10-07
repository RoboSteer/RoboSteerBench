"""Concurrency-safe node-local byte cache for source videos."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


class LocalVideoCache:
    """Materialize source videos byte-for-byte on node-local storage.

    Cache validity follows the evaluator's existing lightweight source
    identity: resolved path, byte size, and nanosecond mtime. A per-entry file
    lock makes publication safe across GPU worker processes.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def materialize(self, source: Path) -> Path:
        source = Path(source).resolve()
        source_stat = source.stat()
        identity = {
            "source": str(source),
            "source_file_size_bytes": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
        }
        digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
        entry_root = self.root / digest[:2]
        entry_root.mkdir(parents=True, exist_ok=True)
        cached = entry_root / f"{digest}{source.suffix.lower()}"
        metadata_path = entry_root / f"{digest}.json"
        lock_path = entry_root / f"{digest}.lock"

        with lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            if self._valid(cached, metadata_path, identity):
                return cached
            temporary = cached.with_name(f".{cached.name}.{os.getpid()}.tmp")
            metadata_temporary = metadata_path.with_name(
                f".{metadata_path.name}.{os.getpid()}.tmp"
            )
            try:
                shutil.copyfile(source, temporary)
                copied_size = temporary.stat().st_size
                current_stat = source.stat()
                if (
                    copied_size != source_stat.st_size
                    or current_stat.st_size != source_stat.st_size
                    or current_stat.st_mtime_ns != source_stat.st_mtime_ns
                ):
                    raise IOError(f"source video changed while caching: {source}")
                os.replace(temporary, cached)
                metadata_temporary.write_text(
                    json.dumps(identity, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(metadata_temporary, metadata_path)
            finally:
                temporary.unlink(missing_ok=True)
                metadata_temporary.unlink(missing_ok=True)
            return cached

    @staticmethod
    def _valid(cached: Path, metadata_path: Path, identity: dict[str, Any]) -> bool:
        if not cached.is_file() or not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return (
                metadata == identity
                and cached.stat().st_size == identity["source_file_size_bytes"]
            )
        except (OSError, ValueError, TypeError):
            return False
