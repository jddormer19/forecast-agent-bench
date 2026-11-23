"""Pre-registered secondary metrics, computed after the one-time holdout scoring (SPEC sections 6 and 10.5).

Log loss, Brier skill against climatology, a reliability table, and the Murphy decomposition of the Brier score
(reliability - resolution + uncertainty) overall and by station, for the primary holdout. Read-only: it reuses the
scorer's loaders and never touches the holdout lock or the primary results.

Run: python -m domains.weather.calibration   ->  domains/weather/calibration_results.json, docs/img/reliability.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.scoring import brier, brier_skill, logloss
from domains.weather.make_readme_charts import BLUE, GREY, INK, MUTED
from domains.weather.score import load_runs

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("calibration_results.json")
IMG = ROOT / "docs" / "img" / "reliability.png"
BINS = 10
SYSTEMS = {"temp": ["A1", "A2", "A3", "ssp", "mos_based", "climatology"],
           "precip": ["A1p", "A2p", "A3p", "ssp", "mos_based", "climatology"]}
LABELS = {"A1": "Agent, all tools", "A1p": "Agent, all tools", "A2": "Agent, no discussion", "A2p": "Agent, no discussion",
          "A3": "Agent, no baseline tool", "A3p": "Agent, no baseline tool", "ssp": "SSP baseline",
          "mos_based": "MOS-based", "climatology": "Climatology"}


def murphy(p, y, bins: int = BINS) -> dict:
    """Binned Murphy decomposition: BS ~= reliability - resolution + uncertainty (+ a small within-bin term)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    ybar, n = y.mean(), len(y)
    rel = res = 0.0
    for k in np.unique(b):
        m = b == k
        nk, pk, ok = m.sum(), p[m].mean(), y[m].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - ybar) ** 2
    rel, res, unc = rel / n, res / n, ybar * (1 - ybar)
    bs = brier(p, y)
    return {"brier": round(bs, 4), "reliability": round(rel, 4), "resolution": round(res, 4),
            "uncertainty": round(unc, 4), "within_bin": round(bs - (rel - res + unc), 4)}


def table(p, y, bins: int = BINS) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for k in range(bins):
        m = b == k
        if m.sum():
            rows.append({"bin": f"{k / bins:.1f}-{(k + 1) / bins:.1f}", "n": int(m.sum()),
                         "mean_forecast": round(float(p[m].mean()), 3), "observed": round(float(y[m].mean()), 3)})
    return rows


def analyse(track: str) -> dict:
    df, _ = load_runs(track)
    df = df[df.period == "holdout_primary"]
    out = {"n_questions": int(len(df)), "base_rate": round(float(df.outcome.mean()), 3), "systems": {}, "by_station": {}}
    for s in SYSTEMS[track]:
        sub = df.dropna(subset=[s])
        y = sub.outcome
        out["systems"][s] = {"label": LABELS[s], "n": int(len(sub)), "log_loss": round(logloss(sub[s], y), 4),
                             "skill_vs_climatology": round(brier_skill(sub[s], y, sub.climatology), 3),
                             **murphy(sub[s], y), "reliability_table": table(sub[s], y)}
    agent = SYSTEMS[track][0]
    for sid, g in df.dropna(subset=[agent, "ssp"]).groupby("sid"):
        out["by_station"][sid] = {"n": int(len(g)), agent: murphy(g[agent], g.outcome), "ssp": murphy(g.ssp, g.outcome)}
    return out


def plot(res: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, track in zip(axes, ["temp", "precip"]):
        agent = SYSTEMS[track][0]
        ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.9)
        for s, color, lw in [(agent, BLUE, 2.2), ("ssp", GREY, 1.6)]:
            rows = [r for r in res[track]["systems"][s]["reliability_table"] if r["n"] >= 10]
            ax.plot([r["mean_forecast"] for r in rows], [r["observed"] for r in rows], "o-", color=color, lw=lw,
                    ms=5, label=f"{LABELS[s]} (rel. {res[track]['systems'][s]['reliability']:.4f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("Forecast probability (bin mean)"); ax.set_ylabel("Observed frequency")
        ax.set_title("Temperature" if track == "temp" else "Precipitation", loc="left", color=INK, fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    IMG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG, dpi=160)


if __name__ == "__main__":
    res = {"note": "Pre-registered secondary metrics (SPEC 6, 10.5), computed 2026-11-22 after the one-time scoring; "
                   "primary results unchanged. Bins of width 0.1; bins with fewer than 10 questions are left out of the plot.",
           "temp": analyse("temp"), "precip": analyse("precip")}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    plot(res)
    for track in ["temp", "precip"]:
        print(f"\n{track}: n={res[track]['n_questions']} base rate {res[track]['base_rate']}")
        print(f"  {'system':26s} {'n':>5} {'Brier':>7} {'logloss':>8} {'skill':>6} {'REL':>7} {'RES':>7} {'UNC':>7}")
        for s, r in res[track]["systems"].items():
            print(f"  {r['label']:26s} {r['n']:5d} {r['brier']:7.4f} {r['log_loss']:8.4f} {r['skill_vs_climatology']:6.3f}"
                  f" {r['reliability']:7.4f} {r['resolution']:7.4f} {r['uncertainty']:7.4f}")
