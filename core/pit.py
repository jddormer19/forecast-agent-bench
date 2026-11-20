"""Point-in-time data access, the heart of the framework.

A domain declares its data as `Source`s. Every row carries `available_at`: the UTC time the information became
knowable (a forecast run's publication, a bulletin's issue time, the end of an observed day). Tools and baselines
never touch raw data; they get a `Snapshot` of every source as of the question's decision time t, so data from
after t is unreachable by construction. The audit log is a second line of defense: every tool call records the
newest `available_at` it was handed, and any value after t is flagged as a leak.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass
class Source:
    name: str
    load: Callable[[], pd.DataFrame]       # must return a frame with a tz-aware UTC `available_at` column
    description: str = ""
    _frame: pd.DataFrame | None = field(default=None, repr=False)

    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            df = self.load()
            if "available_at" not in df:
                raise ValueError(f"source {self.name!r} has no available_at column")
            if df.available_at.dt.tz is None:
                raise ValueError(f"source {self.name!r}: available_at must be timezone-aware UTC")
            self._frame = df.sort_values("available_at", kind="stable").reset_index(drop=True)
        return self._frame


class Snapshot:
    """All sources as of time t. `get` returns only rows with available_at <= t and remembers the newest one."""

    def __init__(self, sources: dict[str, Source], t: pd.Timestamp):
        self.sources, self.t = sources, pd.Timestamp(t)
        self.newest: pd.Timestamp | None = None

    def get(self, name: str) -> pd.DataFrame:
        df = self.sources[name].frame()
        cut = df.available_at.searchsorted(self.t, side="right")   # frame is sorted by available_at
        out = df.iloc[:cut]
        if len(out):
            last = out.available_at.iloc[-1]
            self.newest = last if self.newest is None else max(self.newest, last)
        return out


class ToolBox:
    """Domain tools over point-in-time snapshots, with an audit log of every call.

    A tool is a function `fn(snap: Snapshot, q: dict, **params) -> dict`. `q` is the question row (it must have
    `qid` and `t_utc`). The agent sees only `qid` plus the tool's own params; the time cutoff is never something
    the agent can set.
    """

    def __init__(self, sources: list[Source], questions: pd.DataFrame, log_path: Path | None = None):
        self.sources = {s.name: s for s in sources}
        self.questions = questions.set_index("qid", drop=False)
        self.log_path = log_path
        self.tools: dict[str, Callable] = {}

    def tool(self, fn: Callable) -> Callable:
        params = list(inspect.signature(fn).parameters)
        if params[:2] != ["snap", "q"]:
            raise TypeError(f"tool {fn.__name__} must start with (snap, q)")
        self.tools[fn.__name__] = fn
        return fn

    def snapshot(self, qid: str) -> Snapshot:
        return Snapshot(self.sources, self.questions.loc[qid, "t_utc"])

    def call(self, name: str, qid: str, **params) -> dict:
        if qid not in self.questions.index:
            return {"error": f"unknown question {qid}"}
        snap = self.snapshot(qid)
        result = self.tools[name](snap, self.questions.loc[qid].to_dict(), **params)
        self._log(name, qid, snap, params)
        return result

    def _log(self, name: str, qid: str, snap: Snapshot, params: dict):
        rec = {"logged_at": datetime.now(timezone.utc).isoformat(), "tool": name, "qid": qid, "params": params,
               "t": snap.t.isoformat(), "newest_data": None if snap.newest is None else snap.newest.isoformat(),
               "leak": bool(snap.newest is not None and snap.newest > snap.t)}
        if rec["leak"]:   # unreachable through Snapshot.get; kept as a tripwire for future changes
            raise RuntimeError(f"LEAK: {name} on {qid} saw data from {rec['newest_data']} > t={rec['t']}")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def mcp_server(self, name: str, allowed: list[str] | None = None):
        """Expose the tools as an MCP server (stdio). Each MCP tool takes `qid` plus the tool's own params."""
        from mcp.server.fastmcp import FastMCP   # MCP SDK v1, pinned mcp<2

        server = FastMCP(name)
        for tool_name, fn in self.tools.items():
            if allowed is not None and tool_name not in allowed:
                continue
            server.add_tool(self._mcp_wrapper(tool_name, fn), name=tool_name, description=inspect.getdoc(fn))
        return server

    def _mcp_wrapper(self, tool_name: str, fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        extra = list(sig.parameters.values())[2:]

        def wrapper(qid: str, **params):
            return self.call(tool_name, qid, **params)

        wrapper.__signature__ = sig.replace(parameters=[
            inspect.Parameter("qid", inspect.Parameter.KEYWORD_ONLY, annotation=str), *[
                p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in extra]])
        wrapper.__name__ = tool_name
        return wrapper
