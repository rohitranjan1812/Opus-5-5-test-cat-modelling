"""Concurrent, disk-cached HTTP tile fetcher for the enrichment pipeline.

- Cache: ``$CATFORGE_CACHE_DIR`` (default ``~/.cache/catforge``) ``/tiles/<sha1[:2]>/<sha1>``; a 404 is
  cached as an empty file so missing tiles are not re-requested.
- ``CATFORGE_OFFLINE=1`` serves from the cache only (reproducible runs, air-gapped servers).
- The transport is injectable, so tests never touch the network.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from .. import __version__


def default_cache_dir() -> Path:
    return Path(os.environ.get("CATFORGE_CACHE_DIR") or Path.home() / ".cache" / "catforge")


class TileFetcher:
    def __init__(self, cache_dir: str | Path | None = None, transport: httpx.BaseTransport | None = None,
                 offline: bool | None = None, concurrency: int = 12, timeout: float = 30.0):
        self.root = Path(cache_dir) if cache_dir else default_cache_dir()
        self.offline = offline if offline is not None else os.environ.get("CATFORGE_OFFLINE") == "1"
        self.concurrency = concurrency
        self.client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True,
                                   headers={"User-Agent": f"catforge/{__version__} (exposure enrichment)"})
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "cache_hits": 0, "bytes": 0, "errors": 0, "missing": 0}

    def _count(self, **kw):
        with self._lock:
            for k, v in kw.items():
                self.stats[k] += v

    def _path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.root / "tiles" / h[:2] / h

    def get(self, url: str, cache: bool = True) -> bytes | None:
        p = self._path(url)
        if cache and p.exists():
            self._count(cache_hits=1)
            data = p.read_bytes()
            return data or None
        if self.offline:
            return p.read_bytes() or None if p.exists() else None
        try:
            r = self.client.get(url)
        except httpx.HTTPError:
            self._count(errors=1)
            return p.read_bytes() or None if p.exists() else None  # stale copy beats nothing
        self._count(requests=1, bytes=len(r.content))
        if r.status_code == 404:
            self._count(missing=1)
            self._store(p, b"")
            return None
        if r.status_code != 200:
            self._count(errors=1)
            return None
        self._store(p, r.content)
        return r.content

    def _store(self, p: Path, data: bytes):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def get_many(self, urls: Iterable[str], progress: Callable[[int, int], None] | None = None) -> dict[str, bytes | None]:
        urls = list(dict.fromkeys(urls))
        out: dict[str, bytes | None] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = {ex.submit(self.get, u): u for u in urls}
            for k, f in enumerate(as_completed(futs)):
                out[futs[f]] = f.result()
                if progress:
                    progress(k + 1, len(urls))
        return out
