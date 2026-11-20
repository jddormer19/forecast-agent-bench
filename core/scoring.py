"""Scoring for probability forecasts of binary questions: Brier, log loss, paired bootstrap, reliability."""
from __future__ import annotations

import numpy as np
import pandas as pd


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def logloss(p, y, eps: float = 1e-6) -> float:
    p, y = np.clip(np.asarray(p, float), eps, 1 - eps), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired_bootstrap(a, b, y, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of Brier(a) - Brier(b) on the same questions, with a 95% CI from resampling questions.

    Negative means `a` is better. `a` "beats" `b` only if the whole interval is below 0.
    """
    a, b, y = (np.asarray(x, float) for x in (a, b, y))
    if not (len(a) == len(b) == len(y)) or len(y) == 0:
        raise ValueError("a, b and y must be non-empty and the same length")
    d = (a - y) ** 2 - (b - y) ** 2
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(preds: pd.DataFrame, outcome: str, systems: list[str], reference: str, **boot) -> pd.DataFrame:
    """One row per system: Brier, log loss, and the paired difference vs `reference` on the same rows."""
    rows = []
    for s in systems:
        d, lo, hi = paired_bootstrap(preds[s], preds[reference], preds[outcome], **boot) if s != reference else (0, 0, 0)
        rows.append({"system": s, "n": len(preds), "brier": brier(preds[s], preds[outcome]),
                     "logloss": logloss(preds[s], preds[outcome]), f"minus_{reference}": d,
                     "ci_low": lo, "ci_high": hi, "beats_reference": hi < 0, "worse_than_reference": lo > 0})
    return pd.DataFrame(rows)


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    """Mean forecast vs observed frequency per probability bin (for reliability plots)."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = np.minimum((df.p * bins).astype(int), bins - 1)
    return df.groupby("bin").agg(n=("y", "size"), mean_forecast=("p", "mean"), observed=("y", "mean")).reset_index()
