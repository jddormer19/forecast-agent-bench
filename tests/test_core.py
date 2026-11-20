import json

import numpy as np
import pandas as pd
import pytest

from core.pit import Snapshot, Source, ToolBox
from core.scoring import brier, compare, paired_bootstrap, reliability


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def obs_source():
    return Source("obs", lambda: pd.DataFrame({
        "available_at": [ts("2025-01-01 10:00"), ts("2025-01-01 12:00"), ts("2025-01-01 14:00")],
        "value": [1, 2, 3]}))


def test_snapshot_hides_the_future_and_tracks_newest():
    snap = Snapshot({"obs": obs_source()}, ts("2025-01-01 12:00"))
    assert snap.get("obs").value.tolist() == [1, 2]       # a row available exactly at t is allowed
    assert snap.newest == ts("2025-01-01 12:00")


def test_source_requires_utc_available_at():
    with pytest.raises(ValueError):
        Source("bad", lambda: pd.DataFrame({"x": [1]})).frame()
    with pytest.raises(ValueError):
        Source("bad", lambda: pd.DataFrame({"available_at": [pd.Timestamp("2025-01-01")]})).frame()


def toolbox(tmp_path):
    q = pd.DataFrame({"qid": ["A", "B"], "t_utc": [ts("2025-01-01 11:00"), ts("2025-01-01 15:00")]})
    tb = ToolBox([obs_source()], q, log_path=tmp_path / "calls.jsonl")

    @tb.tool
    def latest(snap, q, n: int = 1) -> dict:
        """Latest n observations."""
        return {"values": snap.get("obs").value.tail(n).tolist()}
    return tb


def test_toolbox_answers_as_of_each_question_and_logs(tmp_path):
    tb = toolbox(tmp_path)
    assert tb.call("latest", "A", n=5) == {"values": [1]}
    assert tb.call("latest", "B", n=5) == {"values": [1, 2, 3]}
    log = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["leak"] for r in log] == [False, False]
    assert log[0]["newest_data"].startswith("2025-01-01T10:00")
    assert tb.call("latest", "nope") == {"error": "unknown question nope"}


def test_tool_signature_is_enforced(tmp_path):
    tb = toolbox(tmp_path)
    with pytest.raises(TypeError):
        tb.tool(lambda q, snap: {})


def test_mcp_server_exposes_qid_and_params_only(tmp_path):
    pytest.importorskip("mcp")
    import asyncio
    server = toolbox(tmp_path).mcp_server("test")
    tools = asyncio.run(server.list_tools())
    schema = tools[0].inputSchema
    assert tools[0].name == "latest" and set(schema["properties"]) == {"qid", "n"}


def test_scoring():
    y = np.array([1, 0, 1, 0])
    assert brier([1, 0, 1, 0], y) == 0 and brier([0.5] * 4, y) == 0.25
    d, lo, hi = paired_bootstrap(np.array([0.9, 0.1, 0.9, 0.1] * 50), np.full(200, 0.5), np.tile(y, 50), n=500)
    assert d < 0 and hi < 0
    t = compare(pd.DataFrame({"y": y, "good": [0.9, 0.1, 0.8, 0.2], "base": 0.5}), "y", ["good", "base"], "base", n=200)
    assert t.set_index("system").loc["base", "minus_base"] == 0
    r = reliability([0.05, 0.95, 0.95], [0, 1, 1], bins=10)
    assert r.set_index("bin").loc[9, "observed"] == 1.0
