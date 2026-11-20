"""Run the weather agent (SPEC 5, 9, 10). All spend goes through one ledger with the $30 hard cap.

  python -m domains.weather.run_agent --track temp --system A1 --period dev --n 10 --seed 101 --label pilot
  python -m domains.weather.run_agent --track temp --system A1 --period holdout_primary --all --label holdout

Systems: A1 (all tools), A2 (no forecast_discussion), A3 (no statistical_baseline); the same names with a "p"
for the precipitation track are accepted too. --model defaults to the pinned claude-haiku-5-5.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agent import Budget, load_env_file, run_questions
from domains.weather.baselines import ssp_for_tools
from domains.weather.build_benchmark import DATA
from domains.weather.tools import build_toolbox

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs"
CAP_USD = 55.0   # raised from $30 on 2026-11-20 at the author's decision (SPEC change log), before any holdout run
PINNED_MODEL = "claude-haiku-5-5"

# get_question is not offered as a tool: its content is the first message (pilot: 14 calls for 10 questions)
ALL_TOOLS = ["model_forecasts", "mos_guidance", "observations_so_far", "recent_weather",
             "model_recent_errors", "forecast_discussion", "statistical_baseline"]
SYSTEMS = {"A1": ALL_TOOLS,
           "A2": [t for t in ALL_TOOLS if t != "forecast_discussion"],
           "A3": [t for t in ALL_TOOLS if t != "statistical_baseline"]}

SYSTEM_PROMPT = {
    "temp": ("You are a weather forecaster. You answer one yes/no question: will today's official high temperature "
             "at a weather station reach a threshold (its normal high for the date)? It is 6 AM local time; every "
             "tool returns only information available now.\n\nHow to work:\n"
             "- Weather-model forecasts are strong but have biases; model_recent_errors shows how each model has "
             "been doing at this station lately.\n"
             "- observations_so_far shows the overnight temperatures; the high usually comes in the afternoon.\n"
             "- The forecast discussion is written by NWS forecasters and may explain fronts, clouds, rain or wind "
             "that change the afternoon high. Use it when it is relevant to this station and today.\n"
             "- statistical_baseline is a well-calibrated statistical model. Move away from it only for a concrete "
             "reason you found, and be calibrated: a forecast that is 1-2°F from the threshold is genuinely uncertain."),
    "precip": ("You are a weather forecaster. You answer one yes/no question: will at least 0.01 inch of "
               "precipitation be recorded at a weather station today? It is 6 AM local time; no measurable "
               "precipitation has fallen yet today; every tool returns only information available now.\n\nHow to work:\n"
               "- Compare the three models' precipitation totals and the official MOS precipitation chances.\n"
               "- The forecast discussion is written by NWS forecasters: timing, coverage and confidence of rain or "
               "storms, and whether they reach this station's area today. Use it when it is relevant.\n"
               "- statistical_baseline is a well-calibrated statistical model. Move away from it only for a concrete "
               "reason you found, and be calibrated: scattered showers make a single station's outcome uncertain."),
}


def pick(questions: pd.DataFrame, period: str, n: int | None, seed: int, scored: set[str]) -> list[str]:
    """Questions from one period; with n, a sample stratified by station (as equal as possible).

    Only questions with a baseline prediction (`scored`) are eligible: on dev that excludes time block 1, which the
    forward-chained baselines use only for training, exactly as dev scoring does.
    """
    q = questions[(questions.period == period) & questions.qid.isin(scored)]
    if n is None:
        return q.qid.tolist()
    per = max(1, n // q.sid.nunique())
    s = q.groupby("sid", group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
    return s.sample(frac=1, random_state=seed).qid.tolist()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["temp", "precip"], required=True)
    ap.add_argument("--system", required=True, help="A1, A2, A3 (or A1p, A2p, A3p)")
    ap.add_argument("--period", choices=["dev", "holdout_primary", "holdout_secondary"], required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=PINNED_MODEL)
    ap.add_argument("--label", required=True, help="run name, e.g. pilot / dev / holdout")
    ap.add_argument("--est-cost", type=float, default=0.005, help="per-question estimate used by the budget check")
    a = ap.parse_args()
    if not a.all and a.n is None:
        ap.error("pass --n or --all")
    system = a.system.rstrip("p")
    load_env_file(ROOT / ".env")
    q = pd.read_parquet(DATA / f"questions_{a.track}.parquet")
    q["date"] = pd.to_datetime(q.date).dt.date
    ssp = ssp_for_tools(a.track)
    tb = build_toolbox(q, a.track, log_path=RUNS / "tool_calls.jsonl", ssp=ssp)
    scored = set(pd.read_parquet(DATA / f"baselines_{a.track}.parquet").qid)
    qids = pick(q, a.period, None if a.all else a.n, a.seed, scored)
    name = f"{a.label}_{a.track}_{a.system}_{a.model.replace('claude-', '')}_{a.period}"
    out = RUNS / f"{name}.jsonl"
    print(f"{name}: {len(qids)} questions -> {out.relative_to(ROOT)}")
    intro = lambda qid: json.dumps(tb.call("get_question", qid), ensure_ascii=False)
    asyncio.run(run_questions(tb, qids, system=SYSTEM_PROMPT[a.track], model=a.model,
                              budget=Budget(RUNS / "spend.json", CAP_USD), run_label=name, out_path=out,
                              allowed=SYSTEMS[system], est_cost_usd=a.est_cost, fallback=ssp, intro=intro))


if __name__ == "__main__":
    main()
