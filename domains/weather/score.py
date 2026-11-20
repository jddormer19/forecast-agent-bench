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


RUNS = DATA.parent.parent / "data" / "runs"


def agent_files(track: str) -> dict[str, list]:
    """System name -> run files (primary + secondary joined; qids never overlap between periods)."""
    s = "" if track == "temp" else "p"
    f = lambda pat: sorted(RUNS.glob(pat))
    return {f"A1{s}": f(f"holdout_{track}_A1{s}_haiku-5-5_holdout_*.jsonl"),
            f"A2{s}": f(f"holdout_{track}_A2{s}_haiku-5-5_holdout_primary.jsonl"),
            f"A3{s}": f(f"holdout_{track}_A3{s}_haiku-5-5_holdout_primary.jsonl"),
            "sonnet": f(f"h8_{track}_A1{s}_sonnet-5-5_holdout_primary.jsonl"),
            "opus": f(f"h8_{track}_A1{s}_opus-5-5_holdout_primary.jsonl")}


def load_runs(track: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(DATA / f"baselines_{track}.parquet")
    meta = {}
    for name, files in agent_files(track).items():
        if not files:
            continue
        r = pd.concat([pd.read_json(p, lines=True) for p in files], ignore_index=True)
        r["error"] = r["error"] if "error" in r else None
        meta[name] = {"files": [p.name for p in files], "answered": len(r), "errors": int(r.error.notna().sum()),
                      "cost_usd": round(float(r.cost_usd.sum()), 4), "usd_per_1000": round(1000 * float(r.cost_usd.mean()), 3),
                      "median_seconds": float(r.seconds.median())}
        df = df.merge(r[["qid", "probability"]].rename(columns={"probability": name}), on="qid", how="left")
    return df, meta


def pair(df: pd.DataFrame, a: str, b: str, label: str, n_boot: int = 10_000) -> dict:
    sub = df.dropna(subset=[a, b])
    d, lo, hi = paired_bootstrap(sub[a], sub[b], sub.outcome, n=n_boot, clusters=sub.date)
    return {"test": label, "a": a, "b": b, "n": len(sub), "brier_a": round(brier(sub[a], sub.outcome), 4),
            "brier_b": round(brier(sub[b], sub.outcome), 4), "a_minus_b": round(d, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "verdict": "a better" if hi < 0 else ("b better" if lo > 0 else "no detectable difference")}


def probe_scores() -> dict:
    p = pd.read_json(RUNS / "probe.jsonl", lines=True)
    p["probability"] = p.output.map(lambda o: o.get("probability") if isinstance(o, dict) else None)
    p[["track", "qid"]] = p.id.str.split("|", expand=True)
    out = {}
    for track in ("temp", "precip"):
        b = pd.read_parquet(DATA / f"baselines_{track}.parquet")[["qid", "period", "outcome", "climatology"]]
        q = pd.read_parquet(DATA / f"questions_{track}.parquet")[["qid", "period", "outcome"]]
        m = p[p.track == track].merge(q, on="qid").dropna(subset=["probability"])
        m = m.merge(b[["qid", "climatology"]], on="qid", how="left")
        out[track] = {per: {"n": int(len(g)), "brier_no_tools": round(brier(g.probability.clip(0, 1), g.outcome), 4),
                            "brier_climatology_same_q": None if g.climatology.isna().all() else
                            round(brier(g.dropna(subset=["climatology"]).climatology,
                                        g.dropna(subset=["climatology"]).outcome), 4)}
                      for per, g in m.groupby("period")}
    return out


def score_holdout_once():
    # Finished scoring -> refused. A run that crashed (only a traceback was shown, no numbers: results print only at
    # the very end) may be retried after a code fix; the retry is recorded in the results.
    prior = LOCK.read_text().strip() if LOCK.exists() else ""
    if prior.startswith("scored"):
        raise SystemExit(f"holdout already scored ({prior}); a second scoring is not allowed")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(f"scoring started {datetime.now(timezone.utc).isoformat()}")
    results, pvals = {"scored_at": datetime.now(timezone.utc).isoformat(),
                      "retry_after_crash": prior or None}, {}
    base = BASELINES + ["extraction"]
    for track in ("temp", "precip"):
        s = "" if track == "temp" else "p"
        A1, A2, A3 = f"A1{s}", f"A2{s}", f"A3{s}"
        df, meta = load_runs(track)
        prim, sec = df[df.period == "holdout_primary"], df[df.period == "holdout_secondary"]
        mid = prim[(prim.ssp >= 0.1) & (prim.ssp <= 0.9)]
        sub100 = prim.dropna(subset=["sonnet", "opus"]) if {"sonnet", "opus"} <= set(prim) else prim.iloc[:0]
        results[track] = {
            "runs": meta,
            "primary_table": table(prim, base + [A1, A2, A3]).to_dict("records"),
            "secondary_table": table(sec, base + [A1]).to_dict("records"),
            "uncertain_subset_table (SSP 0.1-0.9)": table(mid, base + [A1, A2, A3]).to_dict("records"),
            "by_station": {st: pair(g, A1, "ssp", f"{A1} vs SSP at {st}") for st, g in prim.groupby("sid")},
            "pairs": [pair(prim, A1, A2, f"{'H2' if s == '' else 'H5'}: discussions help? ({A1} vs {A2})"),
                      pair(prim, A1, A3, f"{A1} vs {A3} (baseline tool)"),
                      pair(prim, "ssp", "vote", f"{'H3' if s == '' else 'H6'}: SSP vs model vote"),
                      pair(prim, A1, "mos_based", f"{'temp' if s == '' else 'H7'}: {A1} vs MOS-based"),
                      pair(prim, A1, "extraction", f"{A1} vs extraction baseline"),
                      pair(prim, "extraction", "ssp", "extraction vs SSP")],
            "h8_cost_vs_quality (same 100 questions)": [] if sub100.empty else [
                {"model": name, "brier": round(brier(sub100[col], sub100.outcome), 4),
                 "usd_per_1000": meta[col]["usd_per_1000"] if col in meta else None}
                for name, col in (("haiku-5-5", A1), ("sonnet-5-5", "sonnet"), ("opus-5-5", "opus"))]
                + [pair(sub100, "sonnet", A1, "Sonnet vs Haiku"), pair(sub100, "opus", A1, "Opus vs Haiku")],
        }
        sub = prim.dropna(subset=[A1, "ssp"])
        pvals["H1" if s == "" else "H4"] = bootstrap_p(sub[A1], sub.ssp, sub.outcome, clusters=sub.date)
        results[track]["primary_test"] = pair(prim, A1, "ssp", "H1" if s == "" else "H4")
    results["co_primary"] = {"p_values": pvals, "holm_rejected_at_0.05": holm(pvals)}
    results["recall_probe"] = probe_scores()
    (DATA / "holdout" / "results.json").write_text(json.dumps(results, indent=2, default=str))
    LOCK.write_text(f"scored {results['scored_at']}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-once", action="store_true")
    a = ap.parse_args()
    if a.holdout_once:
        score_holdout_once()
    else:
        for tr in ("temp", "precip"):
            score_dev(tr)
