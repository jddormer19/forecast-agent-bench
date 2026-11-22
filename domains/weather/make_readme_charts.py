"""Draw the README's charts from the scored results (docs/img/*.png). Run: python -m domains.weather.make_readme_charts"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "docs" / "img"
DASH = ROOT / "data" / "weather" / "dashboard"
BLUE, GREY, INK, MUTED = "#2563eb", "#9ca3af", "#111827", "#6b7280"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cbd5e1", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def rows(name):
    return json.loads((DASH / f"{name}.json").read_text())["rows"]


def forest():
    keep = ["Agent, all tools", "Agent, no discussion", "Agent, no baseline tool", "NWS MOS-based", "Extraction baseline"]
    data = rows("systems")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for ax, track in zip(axes, ["Precipitation", "Temperature"]):
        rs = [next(r for r in data if r["track"] == track and r["system"] == s) for s in keep]
        for i, r in enumerate(rs[::-1]):
            c = BLUE if r["pre_registered"] else GREY
            ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=c, lw=2.2 if r["pre_registered"] else 1.6)
            ax.plot(r["minus_ssp"], i, "o", color=c, ms=7 if r["pre_registered"] else 5)
        ax.axvline(0, color=MUTED, ls="--", lw=0.8)
        ax.set_yticks(range(len(rs)), [r["system"] for r in rs[::-1]], color=INK)
        ax.set_title(track, loc="left", color=INK, fontweight="bold")
        ax.set_xlabel("Brier difference vs SSP baseline (left = better)")
    fig.tight_layout()
    fig.savefig(IMG / "vs_baseline.png", dpi=160)


def cost_quality():
    data = rows("cost_quality")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, track in zip(axes, ["Temperature", "Precipitation"]):
        rs = sorted([r for r in data if r["track"] == track], key=lambda r: r["usd_per_1000"])
        ax.axhline(rs[0]["ssp_same_questions"], color=MUTED, ls="--", lw=0.9)
        ax.text(1.05, rs[0]["ssp_same_questions"], f"  SSP baseline {rs[0]['ssp_same_questions']:.4f}", color=MUTED,
                va="bottom", fontsize=9)
        top = max(rs, key=lambda r: r["brier"])          # label below the highest point, above all others
        for r in rs:
            ax.plot(r["usd_per_1000"], r["brier"], "o", color=BLUE, ms=7)
            label = r["model"] + "\n" + format(r["brier"], ".4f")
            ax.annotate(label, (r["usd_per_1000"], r["brier"]), textcoords="offset points",
                        xytext=(0, -10 if r is top else 10), ha="center", va="top" if r is top else "bottom",
                        color=INK, fontsize=9, linespacing=1.2)
        ax.set_xscale("log")
        ax.set_xlim(1, 150)
        ax.set_xticks([1, 10, 100])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        lo = min([r["brier"] for r in rs] + [rs[0]["ssp_same_questions"]]); hi = max(r["brier"] for r in rs)
        ax.set_ylim(lo - 0.008, hi + 0.008)
        ax.set_xlabel("$ per 1,000 forecasts (log scale)")
        ax.set_ylabel("Brier (lower = better)")
        ax.set_title(track, loc="left", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / "cost_quality.png", dpi=160)


def harm():
    data = sorted(rows("harm_by_station"), key=lambda r: r["share_of_harm"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh([r["station"] for r in data], [r["share_of_harm"] for r in data],
            color=["#dc2626" if r["share_of_harm"] > 0.3 else GREY for r in data])
    for i, r in enumerate(data):
        ax.text(max(r["share_of_harm"], 0) + 0.01, i, f"{r['share_of_harm']:.0%}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Share of the harm from reading the discussion (temperature)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(IMG / "harm_by_station.png", dpi=160)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    forest(); cost_quality(); harm()
    print(sorted(p.name for p in IMG.glob("*.png")))
