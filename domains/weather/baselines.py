"""Non-LLM baselines (SPEC 5, 9, 10.3). Run: python -m domains.weather.baselines

For each track:
  1. features for every question, computed through point-in-time snapshots (same data the agent tools see);
  2. dev predictions by forward chaining over 6 time blocks (each block predicted by models fit on earlier
     blocks only; block 1 has no prediction);
  3. holdout predictions from models fit on all of dev.
Writes data/weather/baselines_{track}.parquet. Nothing here scores the holdout: see domains/weather/score.py.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from domains.weather import sources as S
from domains.weather.build_benchmark import DATA
from domains.weather.stations import STATIONS
from domains.weather.tools import build_toolbox

BLOCKS = 6
SSP_TEMP = ["mean_minus_thr", "spread", "overnight_max_minus_thr", "yesterday_error", "doy_sin", "doy_cos"]
SSP_PRECIP = ["log1p_gfs_global", "log1p_ecmwf_ifs025", "log1p_icon_global", "n_ge_001", "n_ge_010",
              "doy_sin", "doy_cos"]
STATION_COLS = [f"st_{s.sid}" for s in STATIONS]


def features(track: str, questions: pd.DataFrame) -> pd.DataFrame:
    tb = build_toolbox(questions, track)
    rows = []
    for i, q in enumerate(questions.to_dict("records")):
        snap = tb.snapshot(q["qid"])
        f = S.ssp_temp_features(snap, q) if track == "temp" else S.ssp_precip_features(snap, q)
        mos = S.mos_guidance(snap, q["sid"], q["date"])
        if track == "temp":
            hi = S.model_highs(snap, q["sid"], q["date"])
            f["vote_k"], f["vote_n"] = sum(v >= q["threshold_f"] for v in hi.values()), len(hi)
            f["mos_max_minus_thr"] = (mos["max_temp_f"] - q["threshold_f"]) if mos.get("max_temp_f") is not None \
                else math.nan
        else:
            f["vote_k"], f["vote_n"] = f["n_ge_001"], f["n_models"]
            p06 = [v / 100 for v in (mos.get("p06") or {}).values()]
            f["mos_p06_max"] = max(p06) if p06 else math.nan
            f["mos_p06_combined"] = 1 - float(np.prod([1 - p for p in p06])) if p06 else math.nan
        assert snap.newest is None or snap.newest <= snap.t, f"leak in features for {q['qid']}"
        rows.append({"qid": q["qid"], **f})
        if i % 2000 == 0:
            print(f"{track} features: {i}/{len(questions)}", flush=True)
    return questions.merge(pd.DataFrame(rows), on="qid")


def design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Feature matrix: missing values -> 0 (neutral for these centered features) + station one-hot."""
    x = df[cols].astype(float).fillna(0.0).to_numpy()
    st = np.stack([(df.sid == s.sid).to_numpy(float) for s in STATIONS], axis=1)
    return np.hstack([x, st])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], with_station=True) -> np.ndarray:
    xtr = design(train, cols) if with_station else train[cols].astype(float).fillna(0.0).to_numpy()
    xte = design(test, cols) if with_station else test[cols].astype(float).fillna(0.0).to_numpy()
    m = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, train.outcome.to_numpy())
    return m.predict_proba(xte)[:, 1]


def climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Share of training days with outcome 1, per station and calendar month (shrunk toward the station rate)."""
    tr = train.assign(month=pd.to_datetime(train.date).dt.month)
    st_rate = tr.groupby("sid").outcome.mean()
    g = tr.groupby(["sid", "month"]).outcome.agg(["sum", "count"])
    out = []
    for s, mth in zip(test.sid, pd.to_datetime(test.date).dt.month):
        k, n = g.loc[(s, mth)] if (s, mth) in g.index else (0, 0)
        out.append((k + 5 * st_rate.get(s, 0.5)) / (n + 5))
    return np.array(out)


def predict_all(track: str, df: pd.DataFrame) -> pd.DataFrame:
    ssp_cols = SSP_TEMP if track == "temp" else SSP_PRECIP
    mos_cols = ["mos_max_minus_thr"] if track == "temp" else ["mos_p06_max", "mos_p06_combined"]
    df = df.sort_values(["date", "sid"]).reset_index(drop=True)
    df["vote"] = (df.vote_k + 0.5) / (df.vote_n + 1)
    dev = df[df.period == "dev"].copy()
    dates = np.array(sorted(dev.date.unique()))
    edges = [dates[int(len(dates) * i / BLOCKS)] for i in range(BLOCKS)] + [dates[-1] + pd.Timedelta(days=1)]
    dev["block"] = np.searchsorted(np.array(edges[1:-1]), dev.date.to_numpy(), side="right") + 1
    # the MOS baseline is used only if the archive covers >= 95% of questions (SPEC 9, 10.3)
    mos_cov = float(df[mos_cols[0]].notna().mean())
    parts = []
    for b in range(2, BLOCKS + 1):
        tr, te = dev[dev.block < b], dev[dev.block == b].copy()
        te["climatology"] = climatology(tr, te)
        te["ssp"] = fit_predict(tr, te, ssp_cols)
        te["mos_based"] = fit_predict(tr, te, mos_cols, with_station=False) if mos_cov >= 0.95 else np.nan
        parts.append(te)
    hold = df[df.period != "dev"].copy()
    hold["block"] = 0
    hold["climatology"] = climatology(dev, hold)
    hold["ssp"] = fit_predict(dev, hold, ssp_cols)
    hold["mos_based"] = fit_predict(dev, hold, mos_cols, with_station=False) if mos_cov >= 0.95 else np.nan
    out = pd.concat(parts + [hold], ignore_index=True)
    out.attrs["mos_coverage"] = mos_cov
    print(f"{track}: MOS coverage {mos_cov:.3f} -> MOS-based baseline {'used' if mos_cov >= 0.95 else 'DROPPED'}")
    return out


def ssp_for_tools(track: str):
    """The SSP probability per qid, for the agent's statistical_baseline tool."""
    p = pd.read_parquet(DATA / f"baselines_{track}.parquet").set_index("qid").ssp
    return lambda qid: float(p[qid])


if __name__ == "__main__":
    for track in ("temp", "precip"):
        q = pd.read_parquet(DATA / f"questions_{track}.parquet")
        q["date"] = pd.to_datetime(q.date)
        f = features(track, q.assign(date=q.date.dt.date))
        f["date"] = pd.to_datetime(f.date)
        preds = predict_all(track, f)
        preds["date"] = preds.date.dt.date
        preds.to_parquet(DATA / f"baselines_{track}.parquet", index=False)
        print(f"wrote baselines_{track}.parquet: {len(preds)} rows "
              f"(dev scored blocks 2-{BLOCKS}: {(preds.block >= 2).sum()}, holdout: {(preds.block == 0).sum()})")
