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


def bootstrap_diffs(a, b, y, n: int = 10_000, seed: int = 0, clusters=None) -> tuple[float, np.ndarray]:
    """Mean of Brier(a) - Brier(b) on the same questions and its bootstrap distribution.

    With `clusters` (e.g. the date of each question), whole clusters are resampled together (block bootstrap),
    because questions that share a cluster share weather and are not independent.
    """
    a, b, y = (np.asarray(x, float) for x in (a, b, y))
    if not (len(a) == len(b) == len(y)) or len(y) == 0:
        raise ValueError("a, b and y must be non-empty and the same length")
    d = (a - y) ** 2 - (b - y) ** 2
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    else:
        codes, uniq = pd.factorize(pd.Series(clusters))
        sums, counts = np.bincount(codes, weights=d), np.bincount(codes)
        k = len(uniq)
        means = np.empty(n)
        for i in range(n):
            pick = rng.integers(0, k, k)
            means[i] = sums[pick].sum() / counts[pick].sum()
    return float(d.mean()), means


def paired_bootstrap(a, b, y, n: int = 10_000, seed: int = 0, clusters=None, level: float = 0.95):
    """(mean difference, CI low, CI high). Negative = `a` better; `a` beats `b` only if the CI is below 0."""
    mean, dist = bootstrap_diffs(a, b, y, n, seed, clusters)
    tail = (1 - level) / 2 * 100
    return mean, float(np.percentile(dist, tail)), float(np.percentile(dist, 100 - tail))


def bootstrap_p(a, b, y, n: int = 10_000, seed: int = 0, clusters=None) -> float:
    """Two-sided bootstrap p-value for 'no difference in Brier score'."""
    _, dist = bootstrap_diffs(a, b, y, n, seed, clusters)
    return float(min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean())))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: which hypotheses are rejected at family-wise error rate alpha."""
    order = sorted(pvalues, key=pvalues.get)
    rejected, m = {}, len(order)
    for i, h in enumerate(order):
        ok = pvalues[h] <= alpha / (m - i) and all(rejected.values())
        rejected[h] = ok
    return rejected


def compare(preds: pd.DataFrame, outcome: str, systems: list[str], reference: str, **boot) -> pd.DataFrame:
    """One row per system: Brier, log loss, and the paired difference vs `reference` on the same rows."""
    rows = []
    for s in systems:
        d, lo, hi = paired_bootstrap(preds[s], preds[reference], preds[outcome], **boot) if s != reference else (0, 0, 0)
        rows.append({"system": s, "n": len(preds), "brier": brier(preds[s], preds[outcome]),
                     "logloss": logloss(preds[s], preds[outcome]), f"minus_{reference}": d,
                     "ci_low": lo, "ci_high": hi, "beats_reference": hi < 0, "worse_than_reference": lo > 0})
    return pd.DataFrame(rows)


def brier_skill(p, y, p_ref) -> float:
    """1 - Brier(p) / Brier(reference); positive = better than the reference (e.g. climatology)."""
    return 1 - brier(p, y) / brier(p_ref, y)


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    """Mean forecast vs observed frequency per probability bin (for reliability plots)."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = np.minimum((df.p * bins).astype(int), bins - 1)
    return df.groupby("bin").agg(n=("y", "size"), mean_forecast=("p", "mean"), observed=("y", "mean")).reset_index()
