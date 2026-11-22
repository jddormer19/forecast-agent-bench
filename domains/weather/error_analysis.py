"""Exploratory error analysis of the scored holdout (post-hoc; nothing here changes a scored result).

  python -m domains.weather.error_analysis quant      where and how A1 (with discussions) lost to A2 (without)
  python -m domains.weather.error_analysis traces N    the N questions where the discussion hurt most, with reasoning
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from domains.weather.build_benchmark import DATA
from domains.weather.score import RUNS

KEYWORDS = {"marine layer / low clouds": r"marine layer|low cloud|stratus|fog|overcast|cloud cover",
            "front / pattern change": r"cold front|front(?:al)? passage|trough|upper low|pattern change",
            "rain / storms": r"shower|thunderstorm|rain|precip",
            "uncertainty language": r"uncertain|confidence is low|low confidence|tricky|difficult|spread"}


def runs(track: str, system: str) -> pd.DataFrame:
    s = "" if track == "temp" else "p"
    r = pd.read_json(RUNS / f"holdout_{track}_{system}{s}_haiku-5-5_holdout_primary.jsonl", lines=True)
    return r.set_index("qid")


def frame(track: str = "temp") -> pd.DataFrame:
    b = pd.read_parquet(DATA / f"baselines_{track}.parquet")
    b = b[b.period == "holdout_primary"].set_index("qid")
    a1, a2 = runs(track, "A1"), runs(track, "A2")
    df = b[["sid", "date", "outcome", "ssp"] + (["threshold_f"] if track == "temp" else [])].copy()
    df["a1"], df["a2"] = a1.probability, a2.probability
    for name, col in (("a1", "a1"), ("a2", "a2"), ("ssp", "ssp")):
        df[f"bs_{name}"] = (df[col] - df.outcome) ** 2
    df["harm"] = df.bs_a1 - df.bs_a2                   # > 0: adding the discussion made this answer worse
    df["moved"] = df.a1 - df.a2                        # how the discussion moved the probability
    df["toward_half"] = (df.a1 - 0.5).abs() < (df.a2 - 0.5).abs()
    df["flipped"] = (df.a1 > 0.5) != (df.a2 > 0.5)
    df["month"] = pd.to_datetime(df.date).dt.month
    # the discussion the agent actually read (from its trace) and its reasoning
    df["read_discussion"] = a1.trace.map(lambda t: any(x["tool"] == "forecast_discussion" for x in t))
    df["reasoning"] = a1.forecast.map(lambda f: (f or {}).get("reasoning", ""))
    df["factors"] = a1.forecast.map(lambda f: " | ".join((f or {}).get("key_factors", [])))
    return df


def quant(track: str = "temp"):
    df = frame(track)
    n = len(df)
    print(f"{track}: {n} primary-holdout questions | mean harm (Brier A1 - A2) {df.harm.mean():+.4f}")
    print(f"read the discussion: {df.read_discussion.mean():.0%}")
    big = df[df.harm.abs() > 0.01]
    print(f"\nanswers moved meaningfully by the discussion (|harm| > 0.01): {len(big)} ({len(big)/n:.0%})")
    print(f"  of those, made worse: {(big.harm > 0).mean():.0%}   made better: {(big.harm < 0).mean():.0%}")
    print(f"\nHOW the discussion moved answers (all questions):")
    print(f"  moved toward 50% (hedged): {df.toward_half.mean():.0%}  | flipped side of 50%: {df.flipped.mean():.0%}")
    print(f"  harm when hedged: {df[df.toward_half].harm.sum():+.2f} total | harm when made bolder: "
          f"{df[~df.toward_half].harm.sum():+.2f} total | harm from flips: {df[df.flipped].harm.sum():+.2f}")
    print(f"  mean |A1-0.5| {(df.a1-0.5).abs().mean():.3f} vs |A2-0.5| {(df.a2-0.5).abs().mean():.3f} vs |SSP-0.5| {(df.ssp-0.5).abs().mean():.3f}")
    print("\nby station (total harm, mean shift):")
    print(df.groupby("sid").agg(n=("harm", "size"), total_harm=("harm", "sum"), mean_harm=("harm", "mean"),
                                hedged=("toward_half", "mean")).round(3).sort_values("total_harm", ascending=False).to_string())
    print("\nby month:")
    print(df.groupby("month").agg(n=("harm", "size"), mean_harm=("harm", "mean")).round(4).to_string())
    # what was in the discussions where harm was largest vs smallest
    afd = pd.read_parquet(DATA / "afd.parquet")
    top = df.sort_values("harm", ascending=False)
    print("\nkeyword rates in the agent's reasoning: 100 most-harmed vs 100 most-helped questions")
    for k, pat in KEYWORDS.items():
        worst = top.head(100).reasoning.str.contains(pat, case=False, regex=True).mean()
        best = top.tail(100).reasoning.str.contains(pat, case=False, regex=True).mean()
        print(f"  {k:28s} harmed {worst:.0%}  helped {best:.0%}")
    # calibration of the shift: when the discussion pushed the probability down, how often was the outcome 0?
    down, up = df[df.moved < -0.05], df[df.moved > 0.05]
    print(f"\nwhen the discussion pushed DOWN (n={len(down)}): outcome=0 in {1-down.outcome.mean():.0%} | "
          f"avg A2 {down.a2.mean():.2f} -> A1 {down.a1.mean():.2f}")
    print(f"when the discussion pushed UP   (n={len(up)}): outcome=1 in {up.outcome.mean():.0%} | "
          f"avg A2 {up.a2.mean():.2f} -> A1 {up.a1.mean():.2f}")


def traces(n: int, track: str = "temp"):
    df = frame(track).sort_values("harm", ascending=False).head(n)
    for qid, r in df.iterrows():
        thr = f" threshold {int(r.threshold_f)}F" if "threshold_f" in r else ""
        print(f"\n### {qid}{thr} | outcome {int(r.outcome)} | SSP {r.ssp:.2f} | A2 {r.a2:.2f} -> A1 {r.a1:.2f} | harm {r.harm:+.3f}")
        print("factors:", r.factors[:400])
        print("reasoning:", r.reasoning[:700])


if __name__ == "__main__":
    cmd = sys.argv[1]
    quant() if cmd == "quant" else traces(int(sys.argv[2]))
