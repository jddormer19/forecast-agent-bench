"""Single structured-output calls (no tools), concurrency-limited and charged to the same Budget as the agent."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable

import anthropic

from core.agent import NO_SERVER_FALLBACK, Budget, cost_usd


async def json_calls(items: list[dict], make_prompt: Callable[[dict], tuple[str, str]], schema: dict, *, model: str,
                     budget: Budget, run_label: str, out_path: Path, effort: str = "low", concurrency: int = 4,
                     est_cost_usd: float = 0.001, max_tokens: int = 4000) -> None:
    """For each item (must have an "id"), send one request and append {"id", "output", usage, cost} to out_path.

    Items already in out_path are skipped (resumable). Errors are recorded per item and do not stop the run.
    """
    client = anthropic.AsyncAnthropic()
    done = {json.loads(l)["id"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    extra = {} if model in NO_SERVER_FALLBACK else {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    async def one(it: dict):
        async with sem:
            async with lock:
                budget.check(est_cost_usd * concurrency)   # leave room for the requests already in flight
            system, user = make_prompt(it)
            t0, row = time.time(), {"id": it["id"], "model": model, "run": run_label}
            usage = {}
            try:
                r = await client.beta.messages.create(
                    model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
                    output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}}, **extra)
                usage = {k: getattr(r.usage, k, 0) or 0 for k in
                         ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
                if r.stop_reason == "refusal":
                    raise RuntimeError("refused")
                row["output"] = json.loads(next(b.text for b in r.content if b.type == "text"))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            row.update(usage=usage, cost_usd=round(cost_usd(usage, model), 6), seconds=round(time.time() - t0, 1))
            async with lock:
                budget.add(row["cost_usd"], run_label)
                with open(out_path, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")

    await asyncio.gather(*(one(it) for it in todo))
