"""In-memory state with optional on-disk persistence, and a background job runner."""

from __future__ import annotations

import logging
import pickle
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine.model import AnalysisResult, CatModel
from ..exposure.portfolio import Portfolio

log = logging.getLogger("catforge")


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    result_id: str | None = None
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "progress": round(self.progress, 4),
                "message": self.message, "result_id": self.result_id, "error": self.error,
                "created": self.created, "started": self.started, "finished": self.finished,
                "elapsed_s": round((self.finished or time.time()) - (self.started or self.created), 2)}


class Store:
    def __init__(self, model: CatModel, data_dir: str | Path | None = None, workers: int = 1):
        self.model = model
        self.portfolios: dict[str, Portfolio] = {}
        self.analyses: dict[str, AnalysisResult] = {}
        self.jobs: dict[str, Job] = {}
        self.cache: dict[tuple, Any] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="catforge-job")
        self.data_dir = Path(data_dir) if data_dir else None
        if self.data_dir:
            (self.data_dir / "portfolios").mkdir(parents=True, exist_ok=True)
            (self.data_dir / "analyses").mkdir(parents=True, exist_ok=True)
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self):
        for f in sorted((self.data_dir / "portfolios").glob("*.pkl")):
            try:
                with open(f, "rb") as fh:
                    pf = pickle.load(fh)
                self.portfolios[pf.id] = pf
            except Exception as exc:  # pragma: no cover
                log.warning("could not load %s: %s", f, exc)
        for f in sorted((self.data_dir / "analyses").glob("*.pkl")):
            try:
                with open(f, "rb") as fh:
                    res = pickle.load(fh)
                self.analyses[res.id] = res
            except Exception as exc:  # pragma: no cover
                log.warning("could not load %s: %s", f, exc)

    def _save(self, kind: str, obj):
        if not self.data_dir:
            return
        path = self.data_dir / kind / f"{obj.id}.pkl"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def _delete(self, kind: str, oid: str):
        if self.data_dir:
            (self.data_dir / kind / f"{oid}.pkl").unlink(missing_ok=True)

    # ------------------------------------------------------------------ objects
    def add_portfolio(self, pf: Portfolio) -> Portfolio:
        with self.lock:
            self.portfolios[pf.id] = pf
        self._save("portfolios", pf)
        return pf

    def delete_portfolio(self, pid: str):
        with self.lock:
            self.portfolios.pop(pid)
        self._delete("portfolios", pid)

    def add_analysis(self, res: AnalysisResult) -> AnalysisResult:
        with self.lock:
            self.analyses[res.id] = res
        self._save("analyses", res)
        return res

    def delete_analysis(self, aid: str):
        with self.lock:
            self.analyses.pop(aid)
        self._delete("analyses", aid)

    # ------------------------------------------------------------------ jobs
    def submit(self, kind: str, fn: Callable[[Job], Any]) -> Job:
        job = Job(id="job_" + uuid.uuid4().hex[:10], kind=kind)
        with self.lock:
            self.jobs[job.id] = job

        def run():
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job)
                job.status, job.progress = "done", 1.0
            except Exception as exc:
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                log.error("job %s failed\n%s", job.id, traceback.format_exc())
            finally:
                job.finished = time.time()

        self.executor.submit(run)
        return job

    def cached(self, key: tuple, fn: Callable[[], Any]):
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        val = fn()
        with self.lock:
            self.cache[key] = val
        return val
