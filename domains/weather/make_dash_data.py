"""Write the dashboard's data files from the scored holdout results and the error analysis (read-only inputs).

Run: python -m domains.weather.make_dash_data  ->  data/weather/dashboard/*.json, each {rows, meta}.
"""
import json
from pathlib import Path

import pandas as pd

from domains.weather.build_benchmark import DATA
from domains.weather.error_analysis import KEYWORDS, frame

SRC = Path(__file__).read_text()
OUT = DATA / "dashboard"
RESULTS = json.loads((Path(__file__).parent / "holdout_results.json").read_text())
TRACK = {"temp": "Temperature", "precip": "Precipitation"}
LABEL = {"climatology": "Climatology", "vote": "Model vote", "ssp": "SSP baseline", "mos_based": "NWS MOS-based",
         "extraction": "Extraction baseline", "A1": "Agent, all tools", "A2": "Agent, no discussion",
         "A3": "Agent, no baseline tool"}


def write(name: str, rows: list, inputs: str):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = [{"label": "make_dash_data.py", "value": SRC, "type": "code", "lang": "python"},
            {"label": "Input", "value": inputs}]
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=1))
    print(name, len(rows))


def ci(s: str):
    lo, hi = s.strip("[]").split(",")
    return float(lo), float(hi)


rows = []
for t in ("temp", "precip"):
    for r in RESULTS[t]["primary_table"]:
        key = r["system"].rstrip("p") if r["system"].startswith("A") else r["system"]
        lo, hi = ci(r["ci95"])
        rows.append({"id": f"{t}-{key}", "track": TRACK[t], "system": LABEL[key], "n": r["n"], "brier": r["brier"],
                     "skill": r["skill_vs_climatology"], "minus_ssp": r["minus_ssp"], "ci_low": lo, "ci_high": hi,
                     "pre_registered": key == "A1"})
write("systems", rows, "domains/weather/holdout_results.json (primary_table), scored 2026-11-20")

rows = []
for t, h in (("temp", "H1"), ("precip", "H4")):
    p = RESULTS[t]["primary_test"]
    rows.append({"test": h, "track": TRACK[t], "n": p["n"], "agent_brier": p["brier_a"], "ssp_brier": p["brier_b"],
                 "diff": p["a_minus_b"], "ci_low": p["ci95"][0], "ci_high": p["ci95"][1],
                 "p_value": round(RESULTS["co_primary"]["p_values"][h], 4),
                 "holm_significant": RESULTS["co_primary"]["holm_rejected_at_0.05"][h],
                 "verdict": "Agent better" if p["ci95"][1] < 0 else ("Agent worse" if p["ci95"][0] > 0 else "No difference")})
write("coprimary", rows, "domains/weather/holdout_results.json (co_primary, primary_test)")

rows = []
SSP100 = {"temp": 0.0698, "precip": 0.0938}   # SSP on the same 100 questions (exploratory run, 2026-11-20)
for t in ("temp", "precip"):
    for h in RESULTS[t]["h8_cost_vs_quality (same 100 questions)"]:
        if "model" in h:
            rows.append({"id": f"{t}-{h['model']}", "track": TRACK[t], "model": h["model"].replace("-5-5", " 5.5").title(),
                         "brier": h["brier"], "usd_per_1000": h["usd_per_1000"], "ssp_same_questions": SSP100[t]})
write("cost_quality", rows, "holdout_results.json (h8) + SSP on the same 100 questions")

rows = []
for t in ("temp", "precip"):
    for sid, p in RESULTS[t]["by_station"].items():
        rows.append({"id": f"{t}-{sid}", "track": TRACK[t], "station": sid, "n": p["n"], "diff": p["a_minus_b"],
                     "ci_low": p["ci95"][0], "ci_high": p["ci95"][1]})
write("stations", rows, "holdout_results.json (by_station): agent minus SSP")

df = frame("temp")
g = df.groupby("sid").agg(n=("harm", "size"), total_harm=("harm", "sum"), mean_harm=("harm", "mean")).reset_index()
write("harm_by_station", [{"station": r.sid, "n": int(r.n), "total_harm": round(r.total_harm, 3),
                           "mean_harm": round(r.mean_harm, 4), "share_of_harm": round(r.total_harm / df.harm.sum(), 3)}
                          for r in g.sort_values("total_harm", ascending=False).itertuples()],
      "data/runs holdout A1 and A2 (temperature, primary holdout) + baselines_temp.parquet")

down, up = df[df.a1 - df.a2 < -0.05], df[df.a1 - df.a2 > 0.05]
write("pushes", [
    {"direction": "Pushed down", "n": len(down), "went_wrong": round(float(down.outcome.mean()), 3),
     "before": round(float(down.a2.mean()), 2), "after": round(float(down.a1.mean()), 2)},
    {"direction": "Pushed up", "n": len(up), "went_wrong": round(float(1 - up.outcome.mean()), 3),
     "before": round(float(up.a2.mean()), 2), "after": round(float(up.a1.mean()), 2)}],
    "temperature, primary holdout: A1 vs A2 probability moved by more than 0.05")

text = (df.factors + " " + df.reasoning).str.lower()
MODES = {"Area-wide 'below/above normal' statement": r"(?:below|under|above|cooler than|warmer than) normal|degrees? (?:below|under|above) normal",
         "Marine layer, low clouds, fog": r"marine layer|stratus|low cloud|fog|eddy|drizzle",
         "Trusts the text over the models": r"weight the forecaster|cannot read|does not (?:reflect|account|read)|baseline.*(?:can.?t|cannot|does not)",
         "Fronts, clouds, storms": r"cold front|front|cloud|shower|storm|convection|rain"}
top, bottom = df.sort_values("harm", ascending=False).head(100), df.sort_values("harm").head(100)
write("failure_modes", [{"language": k, "most_harmed": round(float(text[top.index].str.contains(p, regex=True).mean()), 2),
                         "most_helped": round(float(text[bottom.index].str.contains(p, regex=True).mean()), 2),
                         "all": round(float(text.str.contains(p, regex=True).mean()), 2)} for k, p in MODES.items()],
      "keyword match on the agent's stated factors and reasoning (temperature, primary holdout)")

rows = []
for t in ("temp", "precip"):
    for per, v in RESULTS["recall_probe"][t].items():
        rows.append({"id": f"{t}-{per}", "track": TRACK[t], "period": "Before cutoff (dev)" if per == "dev" else "After cutoff (holdout)",
                     "n": v["n"], "brier_no_tools": v["brier_no_tools"], "brier_climatology": v["brier_climatology_same_q"]})
write("probe", rows, "holdout_results.json (recall_probe)")
