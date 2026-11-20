"""Domain-agnostic forecasting agent (LangGraph: research -> forecast) with a hard spending cap.

research: the model calls the domain's point-in-time tools in a loop (at most MAX_TOOL_CALLS).
forecast: one final call with structured output (JSON schema) -> probability, confidence, key factors, reasoning.
The tools run in-process through the ToolBox (the same functions can also be served over MCP). Every question
records tokens, dollars, seconds and the tool trace, and a Budget refuses to start a question that could take
total spend past the cap.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

from core.pit import ToolBox

MAX_TOOL_CALLS = 8
# $ per 1M tokens: input, output, cache read (cache writes cost 1.25x input). Haiku's price is for prompts of up
# to 100K tokens; the agent's prompts stay far below that.
PRICES = {"claude-opus-5-5": (4.00, 20.00, 0.20), "claude-sonnet-5-5": (2.00, 10.00, 0.10),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}
NO_SERVER_FALLBACK = {"claude-haiku-5-5"}   # Haiku 5.5 has no server-side fallback; don't send the beta for it

FORECAST_SCHEMA = {
    "type": "object",
    "properties": {
        "probability": {"type": "number", "description": "Probability that the answer is YES, between 0 and 1"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["probability", "confidence", "key_factors", "reasoning"],
    "additionalProperties": False,
}
FORECAST_PROMPT = ("Give your final forecast now: the probability that the answer is YES, your confidence "
                   "(low/medium/high), the key factors, and short reasoning.")


def cost_usd(usage: dict, model: str) -> float:
    pin, pout, pcache = PRICES[model]
    return (usage.get("input_tokens", 0) * pin + usage.get("cache_creation_input_tokens", 0) * pin * 1.25
            + usage.get("cache_read_input_tokens", 0) * pcache + usage.get("output_tokens", 0) * pout) / 1e6


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Cumulative spend across every run, stored in a ledger file so separate runs share one cap.

    Reads and updates take an exclusive file lock, so two processes running at once cannot lose each other's
    updates (a lost update would undercount spend and weaken the cap).
    """
    ledger: Path
    cap_usd: float

    def _locked(self):
        import fcntl
        from contextlib import contextmanager

        @contextmanager
        def cm():
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger.with_suffix(".lock"), "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lk, fcntl.LOCK_UN)
        return cm()

    def _read(self) -> dict:
        return json.loads(self.ledger.read_text()) if self.ledger.exists() else {"spent_usd": 0.0, "by_run": {}}

    def spent(self) -> float:
        with self._locked():
            return self._read()["spent_usd"]

    def check(self, next_estimate_usd: float):
        spent = self.spent()
        if spent + next_estimate_usd > self.cap_usd:
            raise BudgetExceeded(f"spent ${spent:.2f}; next request (~${next_estimate_usd:.3f}) would pass "
                                 f"the ${self.cap_usd:.2f} cap")

    def add(self, usd: float, label: str):
        with self._locked():
            data = self._read()
            data["spent_usd"] = round(data["spent_usd"] + usd, 6)
            data["by_run"][label] = round(data["by_run"].get(label, 0.0) + usd, 6)
            tmp = self.ledger.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.ledger)


class AgentState(TypedDict):
    qid: str
    messages: list
    tool_calls: int
    trace: list
    usage: dict
    forecast: dict | None


def _add_usage(usage: dict, resp) -> dict:
    out = dict(usage)
    for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        out[k] = out.get(k, 0) + (getattr(resp.usage, k, 0) or 0)
    out["requests"] = out.get("requests", 0) + 1
    return out


def tool_specs(toolbox: ToolBox, allowed: list[str] | None) -> list[dict]:
    """Anthropic tool definitions from the ToolBox. The agent never passes qid: the runner adds it."""
    import inspect
    import typing
    specs = []
    for name, fn in toolbox.tools.items():
        if allowed is not None and name not in allowed:
            continue
        hints = typing.get_type_hints(fn)   # resolves string annotations (from __future__ import annotations)
        props = {}
        for p in list(inspect.signature(fn).parameters.values())[2:]:
            h = hints.get(p.name, str)
            if typing.get_origin(h) is typing.Literal:
                props[p.name] = {"type": "string", "enum": list(typing.get_args(h))}
            else:
                props[p.name] = {"type": {int: "integer", float: "number", bool: "boolean"}.get(h, "string")}
            if p.default is not inspect.Parameter.empty:
                props[p.name]["default"] = p.default
        specs.append({"name": name, "description": inspect.getdoc(fn) or "",
                      "input_schema": {"type": "object", "properties": props, "additionalProperties": False}})
    return specs


def build_graph(client: anthropic.AsyncAnthropic, toolbox: ToolBox, tools: list[dict], system: str,
                model: str, effort: str):
    names = {t["name"] for t in tools}
    common: dict[str, Any] = dict(model=model, max_tokens=16000, system=system, tools=tools,
                                  cache_control={"type": "ephemeral"})
    if model not in NO_SERVER_FALLBACK:
        common.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

    async def research(state: AgentState) -> dict:
        msgs, calls, trace, usage = list(state["messages"]), state["tool_calls"], list(state["trace"]), dict(state["usage"])
        while True:
            resp = await client.beta.messages.create(messages=msgs, output_config={"effort": effort}, **common)
            usage = _add_usage(usage, resp)
            if resp.stop_reason == "refusal":
                raise RuntimeError("refused")
            msgs.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                break
            results = []
            for b in uses:
                if calls >= MAX_TOOL_CALLS:
                    content, is_err = "Tool budget reached. Give your forecast with what you have.", True
                elif b.name not in names:
                    content, is_err = f"Tool {b.name} is not available in this run.", True
                else:
                    try:
                        content, is_err = json.dumps(toolbox.call(b.name, state["qid"], **(b.input or {})),
                                                     default=str), False
                    except TypeError as e:    # bad arguments from the model
                        content, is_err = f"Invalid arguments: {e}", True
                    except Exception as e:    # a tool failure goes back to the agent, not up to the runner
                        content, is_err = f"Tool error: {type(e).__name__}", True
                    calls += 1
                    trace.append({"tool": b.name, "input": b.input, "error": is_err})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": is_err})
            msgs.append({"role": "user", "content": results})
            if calls >= MAX_TOOL_CALLS and all(r["is_error"] for r in results):
                break
        return {"messages": msgs, "tool_calls": calls, "trace": trace, "usage": usage}

    async def forecast(state: AgentState) -> dict:
        resp = await client.beta.messages.create(
            messages=state["messages"] + [{"role": "user", "content": FORECAST_PROMPT}], tool_choice={"type": "none"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": FORECAST_SCHEMA}}, **common)
        usage = _add_usage(state["usage"], resp)
        if resp.stop_reason == "refusal":
            raise RuntimeError("refused")
        fc = json.loads(next(b.text for b in resp.content if b.type == "text"))
        fc["probability"] = min(max(float(fc["probability"]), 0.01), 0.99)
        return {"forecast": fc, "usage": usage}

    g = StateGraph(AgentState)
    g.add_node("research", research)
    g.add_node("forecast", forecast)
    g.add_edge(START, "research")
    g.add_edge("research", "forecast")
    g.add_edge("forecast", END)
    return g.compile()


def load_env_file(path: Path) -> None:
    """KEY=VALUE lines into the environment (the project .env is git-ignored and chmod 600)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


async def run_questions(toolbox: ToolBox, qids: list[str], *, system: str, model: str, budget: Budget,
                        run_label: str, out_path: Path, allowed: list[str] | None = None, effort: str = "medium",
                        est_cost_usd: float = 0.01, fallback=None, intro=None) -> list[dict]:
    """Run the agent on each question, appending one JSON line per question to out_path.

    Questions already in out_path are skipped (resumable). A failed question records the error and, if given,
    `fallback(qid)` as its probability (SPEC 10.6). Stops with BudgetExceeded before a question that could pass the cap.
    """
    client = anthropic.AsyncAnthropic()
    tools = tool_specs(toolbox, allowed)
    graph = build_graph(client, toolbox, tools, system, model, effort)
    done = {json.loads(l)["qid"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid in qids:
        if qid in done:
            continue
        budget.check(est_cost_usd)
        t0 = time.time()
        row: dict[str, Any] = {"qid": qid, "model": model, "effort": effort, "run": run_label,
                               "tools": [t["name"] for t in tools]}
        usage: dict = {}
        try:
            st = await graph.ainvoke({"qid": qid, "tool_calls": 0, "trace": [], "usage": {}, "forecast": None,
                                      "messages": [{"role": "user", "content":
                                                    (intro(qid) + "\n\n" if intro else "") +
                                                    f"Answer question {qid}. Use the tools, then stop."}]})
            usage = st["usage"]
            row.update(probability=st["forecast"]["probability"], forecast=st["forecast"],
                       tool_calls=st["tool_calls"], trace=st["trace"])
        except Exception as e:   # one bad question must not kill a run; it is scored with the fallback
            fb = None
            if fallback is not None:
                try:
                    fb = float(fallback(qid))
                    fb = None if fb != fb else fb    # NaN -> None
                except Exception:
                    fb = None
            row.update(error=f"{type(e).__name__}: {e}", probability=fb, used_fallback=fb is not None)
        row.update(usage=usage, cost_usd=round(cost_usd(usage, model), 6), seconds=round(time.time() - t0, 1))
        budget.add(row["cost_usd"], run_label)
        with open(out_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("qid", "probability", "tool_calls", "cost_usd", "seconds", "error")}),
              flush=True)
    return rows
