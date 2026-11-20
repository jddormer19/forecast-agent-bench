"""LLM baselines without tools (SPEC 10.2, 10.3).

  python -m domains.weather.llm_baselines extract   E / Ep: read the latest discussion, extract stated numbers
  python -m domains.weather.llm_baselines fit       calibrate E / Ep on the dev sample -> baselines_*.parquet columns
  python -m domains.weather.llm_baselines probe     R: no tools, no data; how much does the model remember?
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core.agent import Budget, load_env_file
from core.llm import json_calls
from domains.weather import sources as S
from domains.weather.build_benchmark import DATA
from domains.weather.run_agent import CAP_USD, PINNED_MODEL, ROOT, RUNS
from domains.weather.stations import BY_SID
from domains.weather.tools import build_toolbox

DEV_SAMPLE = 1500          # station-days used to calibrate E / Ep (SPEC 10.3)
PROBE_N = 200              # per track and period (SPEC 10.2)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "high_temp_f": {"type": ["number", "null"],
                        "description": "Today's high temperature the discussion states or clearly implies for the "
                                       "station's area, in °F; a range -> its midpoint; null if not stated"},
        "precip_chance_pct": {"type": ["number", "null"],
                              "description": "Today's chance of measurable precipitation the discussion states for "
                                             "the station's area, in %; null if not stated"},
        "quote": {"type": "string", "description": "The sentence(s) the numbers come from, or empty"},
    },
    "required": ["high_temp_f", "precip_chance_pct", "quote"],
    "additionalProperties": False,
}
EXTRACT_SYSTEM = ("You extract numbers from an NWS Area Forecast Discussion. Do not forecast and do not reason about "
                  "the weather: report only what the text states (or a range it states) for the given station's area "
                  "and the given local date. If the text does not state a value, return null.")

PROBE_SCHEMA = {"type": "object", "properties": {"probability": {"type": "number"}},
                "required": ["probability"], "additionalProperties": False}
PROBE_SYSTEM = ("Answer from your own knowledge only. You have no tools and no data. Give the probability that the "
                "statement is true, between 0 and 1.")


def station_days() -> pd.DataFrame:
    """Every (station, date) that appears in either track, with its period and decision time."""
    frames = []
    for track in ("temp", "precip"):
        q = pd.read_parquet(DATA / f"questions_{track}.parquet")
        frames.append(q[["sid", "date", "t_utc", "period"]])
    sd = pd.concat(frames).drop_duplicates(["sid", "date"]).reset_index(drop=True)
    sd["date"] = pd.to_datetime(sd.date).dt.date
    sd["id"] = sd.sid + "-" + sd.date.astype(str)
    return sd


def extract_items() -> list[dict]:
    sd = station_days()
    dev = sd[sd.period == "dev"].sample(DEV_SAMPLE, random_state=21)
    use = pd.concat([dev, sd[sd.period != "dev"]])
    tb = build_toolbox(use.rename(columns={"id": "qid"}).assign(threshold_f=None), "temp")
    items = []
    for r in use.itertuples():
        snap = tb.snapshot(r.id)
        d = S.latest_discussions(snap, BY_SID[r.sid].office, 1)
        assert snap.newest is None or snap.newest <= snap.t
        if d.empty:
            continue
        st = BY_SID[r.sid]
        items.append({"id": r.id, "sid": r.sid, "date": str(r.date), "city": st.city,
                      "issued": d.issued_utc.iloc[0].strftime("%Y-%m-%d %H:%MZ"), "text": d.text.iloc[0][:12000]})
    return items


def extract_prompt(it: dict) -> tuple[str, str]:
    return EXTRACT_SYSTEM, (f"Station: {it['city']} ({it['sid']}). Local date: {it['date']}. "
                            f"Discussion issued {it['issued']}:\n\n{it['text']}")


def probe_items() -> list[dict]:
    items = []
    for track, seed in (("temp", 31), ("precip", 32)):
        q = pd.read_parquet(DATA / f"questions_{track}.parquet")
        for period in ("dev", "holdout_primary"):
            s = q[q.period == period].sample(PROBE_N, random_state=seed)
            for r in s.itertuples():
                st = BY_SID[r.sid]
                stmt = (f"The official high temperature at {st.city} ({st.sid}) on {r.date} was at least "
                        f"{r.threshold_f}°F." if track == "temp" else
                        f"At least 0.01 inch of precipitation was recorded at {st.city} ({st.sid}) on {r.date}.")
                items.append({"id": f"{track}|{r.qid}", "statement": stmt})
    return items


def run(kind: str):
    load_env_file(ROOT / ".env")
    budget = Budget(RUNS / "spend.json", CAP_USD)
    if kind == "extract":
        items = extract_items()
        print(f"extract: {len(items)} station-days")
        asyncio.run(json_calls(items, extract_prompt, EXTRACT_SCHEMA, model=PINNED_MODEL, budget=budget,
                               run_label="extract", out_path=RUNS / "extract.jsonl", est_cost_usd=0.0006))
    else:
        items = probe_items()
        print(f"probe: {len(items)} statements")
        asyncio.run(json_calls(items, lambda it: (PROBE_SYSTEM, it["statement"]), PROBE_SCHEMA, model=PINNED_MODEL,
                               budget=budget, run_label="probe", out_path=RUNS / "probe.jsonl", est_cost_usd=0.0003,
                               max_tokens=2000))


def fit():
    """Calibrate E / Ep on the dev sample (SPEC 10.3) and add `extraction` columns to the baseline files."""
    ex = pd.read_json(RUNS / "extract.jsonl", lines=True)
    ex = ex[ex.output.notna()]
    ex["stated_high"] = ex.output.map(lambda o: o.get("high_temp_f"))
    ex["stated_pop"] = ex.output.map(lambda o: o.get("precip_chance_pct"))
    ex = ex.set_index("id")[["stated_high", "stated_pop"]]
    for track in ("temp", "precip"):
        b = pd.read_parquet(DATA / f"baselines_{track}.parquet")
        b["id"] = b.qid
        b = b.join(ex, on="id")
        if track == "temp":
            b["x"] = b.stated_high.astype(float) - b.threshold_f
        else:
            p = (b.stated_pop.astype(float) / 100).clip(0.01, 0.99)
            b["x"] = np.log(p / (1 - p))
        has = b.x.notna()
        cal = b[(b.period == "dev") & has]
        m = LogisticRegression(max_iter=1000).fit(cal[["x"]], cal.outcome)
        b["extraction"] = b.ssp
        b.loc[has, "extraction"] = m.predict_proba(b.loc[has, ["x"]])[:, 1]
        b["extraction_has_value"] = has
        b.drop(columns=["id", "x"]).to_parquet(DATA / f"baselines_{track}.parquet", index=False)
        print(f"{track}: calibrated on {len(cal)} dev questions with a stated value; "
              f"coverage dev {has[b.period == 'dev'].mean():.2f}, holdout {has[b.period != 'dev'].mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["extract", "fit", "probe"])
    w = ap.parse_args().what
    fit() if w == "fit" else run(w)
