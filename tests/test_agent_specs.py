from __future__ import annotations

from typing import Literal

import pandas as pd

from core.agent import Budget, cost_usd, tool_specs
from core.pit import Source, ToolBox


def toolbox():
    q = pd.DataFrame({"qid": ["A"], "t_utc": [pd.Timestamp("2025-01-01", tz="UTC")]})
    tb = ToolBox([Source("s", lambda: pd.DataFrame({"available_at": [pd.Timestamp("2025-01-01", tz="UTC")]}))], q)

    @tb.tool
    def plain(snap, q) -> dict:
        """No params."""
        return {}

    @tb.tool
    def with_params(snap, q, days: int = 5, which: Literal["latest", "previous"] = "latest") -> dict:
        """Two params."""
        return {}
    return tb


def test_tool_specs_resolve_string_annotations():
    specs = {s["name"]: s for s in tool_specs(toolbox(), None)}
    props = specs["with_params"]["input_schema"]["properties"]
    assert props["days"] == {"type": "integer", "default": 5}
    assert props["which"] == {"type": "string", "enum": ["latest", "previous"], "default": "latest"}
    assert specs["plain"]["input_schema"]["properties"] == {}
    assert "qid" not in props


def test_allowlist():
    assert [s["name"] for s in tool_specs(toolbox(), ["plain"])] == ["plain"]


def test_budget_blocks_before_cap(tmp_path):
    import pytest
    from core.agent import BudgetExceeded
    b = Budget(tmp_path / "spend.json", cap_usd=0.01)
    b.add(0.008, "run1")
    b.check(0.001)
    with pytest.raises(BudgetExceeded):
        b.check(0.005)
    assert round(b.spent(), 3) == 0.008


def test_budget_survives_concurrent_writers(tmp_path):
    from multiprocessing import Process
    ledger = tmp_path / "spend.json"
    procs = [Process(target=_add_many, args=(ledger,)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert round(Budget(ledger, 100).spent(), 6) == round(4 * 50 * 0.001, 6)   # no lost updates


def _add_many(ledger):
    b = Budget(ledger, 100)
    for _ in range(50):
        b.add(0.001, "x")


def test_cost():
    assert abs(cost_usd({"input_tokens": 1_000_000, "output_tokens": 0}, "claude-haiku-5-5") - 0.10) < 1e-9
