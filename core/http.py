"""Polite HTTP for data downloads: per-source throttle, on-disk cache, retry with backoff.

Every provider gets its own `CachedClient` with its published rate limit. A request that was fetched once is
read from the cache forever after, so re-running a pipeline never hits the provider again.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import requests

USER_AGENT = "forecast-agent-bench/0.1"  # generic on purpose: no personal email (data rules, SPEC section 7)
RETRY_STATUS = {429, 500, 502, 503, 504}


class CachedClient:
    def __init__(self, cache_dir: Path, min_interval_s: float, max_retries: int = 5, timeout_s: float = 60,
                 session: requests.Session | None = None, sleep=time.sleep, clock=time.monotonic):
        self.cache_dir = Path(cache_dir)
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._sleep, self._clock = sleep, clock
        self._last = None
        self.network_calls = 0

    def _key(self, method: str, url: str, params, body) -> Path:
        raw = json.dumps([method, url, params, body], sort_keys=True, default=str).encode()
        return self.cache_dir / f"{hashlib.sha256(raw).hexdigest()[:32]}.gz"

    def _throttle(self):
        if self._last is not None:
            wait = self.min_interval_s - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def fetch(self, url: str, params: dict | None = None, json_body: dict | None = None) -> str:
        """Return the response text, from cache when present. Raises after `max_retries` failed attempts."""
        method = "POST" if json_body is not None else "GET"
        path = self._key(method, url, params, json_body)
        if path.exists():
            return gzip.decompress(path.read_bytes()).decode("utf-8")
        backoff = max(self.min_interval_s, 2.0)
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.network_calls += 1
            try:
                r = self.session.request(method, url, params=params, json=json_body, timeout=self.timeout_s)
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                self._sleep(backoff); backoff *= 2
                continue
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                self._sleep(max(backoff, float(r.headers.get("Retry-After", 0) or 0))); backoff *= 2
                continue
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(r.text.encode("utf-8")))
            tmp.replace(path)  # atomic: an interrupted run never leaves a half-written cache file
            return r.text
        raise RuntimeError("unreachable")
