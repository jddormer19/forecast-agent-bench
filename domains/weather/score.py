"""Score systems on the weather benchmark (SPEC 6, 9, 10.5).

  python -m domains.weather.score                      dev only (forward-chained blocks 2-6): free to re-run
  python -m domains.weather.score --holdout-once       the one-time holdout scoring: refuses to run twice

Every comparison is paired on the same questions, with a date-block bootstrap (10,000 resamples).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from core.scoring import bootstrap_p, brier, brier_skill, holm, paired_bootstrap
from domains.weather.build_benchmark import DATA

LOCK = DATA / "holdout" / "SCORED.lock"
BASELINES = ["climatology", "vote", "ssp", "mos_based"]


def table(df: pd.DataFrame, systems: list[str], reference: str = "ssp", n_boot: int = 10_000) -> pd.DataFrame:
    rows = []
    for s in systems:
        sub = df.dropna(subset=[s, reference])
        if sub.empty:
            continue
        d, lo, hi = paired_bootstrap(sub[s], sub[reference], sub.outcome, n=n_boot, clusters=sub.date) \
            if s != reference else (0.0, 0.0, 0.0)
        rows.append({"system": s, "n": len(sub), "brier": round(brier(sub[s], sub.outcome), 4),
                     "skill_vs_climatology": round(brier_skill(sub[s], sub.outcome, sub.climatology), 3),
                     f"minus_{reference}": round(d, 4), "ci95": f"[{lo:+.4f}, {hi:+.4f}]"})
    return pd.DataFrame(rows)


def load(track: str, agent_runs: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"baselines_{track}.parquet")
    for name, path in (agent_runs or {}).items():
        runs = pd.read_json(path, lines=True)[["qid", "probability"]].rename(columns={"probability": name})
        df = df.merge(runs, on="qid", how="left")
    return df


def score_dev(track: str):
    df = load(track)
    dev = df[(df.period == "dev") & (df.block >= 2)]
    print(f"\n{track.upper()} dev (forward-chained blocks 2-6), date-block bootstrap:")
    print(table(dev, [s for s in BASELINES if dev[s].notna().any()]).to_string(index=False))


def score_holdout_once(agent_runs: dict[str, dict[str, str]]):
    if LOCK.exists():
        raise SystemExit(f"holdout already scored ({LOCK.read_text().strip()}); a second scoring is not allowed")
    results, pvals = {}, {}
    for track, primary_system in (("temp", "A1"), ("precip", "A1p")):
        df = load(track, agent_runs.get(track))
        prim = df[df.period == "holdout_primary"]
        results[track] = {"primary": table(prim, BASELINES + list(agent_runs.get(track, {}))).to_dict("records"),
                          "secondary": table(df[df.period == "holdout_secondary"],
                                             BASELINES + list(agent_runs.get(track, {}))).to_dict("records")}
        if primary_system in prim:
            sub = prim.dropna(subset=[primary_system, "ssp"])
            pvals["H1" if track == "temp" else "H4"] = bootstrap_p(sub[primary_system], sub.ssp, sub.outcome,
                                                                    clusters=sub.date)
    results["co_primary"] = {"p_values": pvals, "holm_rejected_at_0.05": holm(pvals)} if pvals else {}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    (DATA / "holdout" / "results.json").write_text(json.dumps(results, indent=2, default=str))
    LOCK.write_text(f"scored {datetime.now(timezone.utc).isoformat()}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-once", action="store_true")
    ap.add_argument("--runs", default="{}", help='JSON: {"temp": {"A1": "path.jsonl", ...}, "precip": {...}}')
    a = ap.parse_args()
    if a.holdout_once:
        score_holdout_once(json.loads(a.runs))
    else:
        for tr in ("temp", "precip"):
            score_dev(tr)
