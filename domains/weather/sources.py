"""Weather data as point-in-time Sources (SPEC sections 4, 9, 10) and the features built from them.

Each source states when every row became knowable (`available_at`, UTC):
  daily      ACIS high / precipitation for a climate day   end of that climate day + 1 h
  fc_temp    Open-Meteo previous_dayN hourly temperature    valid time - N days + FORECAST_LATENCY
  fc_precip  Open-Meteo previous_dayN hourly precipitation  same rule
  asos       hourly station observations                   observation time
  afd        NWS Area Forecast Discussions                 issue time
  mos        NWS GFS MOS guidance                          run time + MOS_LATENCY

Features are computed only through `Snapshot.get`, so baselines and agent tools see exactly the same data.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.pit import Snapshot, Source
from domains.weather.build_benchmark import DATA, climate_day_start_utc, parse_int, parse_pcpn
from domains.weather.download import MODELS
from domains.weather.stations import BY_SID

# A previous_dayN value comes from a run initialized about N days before its valid time; runs are published a few
# hours after initialization. The leak check (SPEC 10.7) verifies this rule against exact runs.
FORECAST_LATENCY = pd.Timedelta(hours=6)
MOS_LATENCY = pd.Timedelta(hours=4)
TEMP_HOURS = range(0, 21)          # local clock hours used for the forecast high (SPEC 4)
MEASURABLE_IN = 0.01


def _daily() -> pd.DataFrame:
    a = pd.read_parquet(DATA / "acis.parquet")
    a["date"] = pd.to_datetime(a.date).dt.date
    a["maxt"] = a.maxt_raw.map(parse_int).astype(float)
    a["pcpn"] = a.pcpn_raw.map(parse_pcpn).astype(float)
    a["available_at"] = [climate_day_start_utc(s, d + timedelta(days=1)) + pd.Timedelta(hours=1)
                         for s, d in zip(a.sid, a.date)]
    return a[["sid", "date", "maxt", "pcpn", "available_at"]]


def usable_forecasts(f: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Drop forecast values whose source run cannot be verified (SPEC change log, 2026-11-19 leak check).

    ECMWF temperature is produced every 3 hours; Open-Meteo fills the hours between by interpolation, and those
    interpolated values could not be traced to a model run. Only ECMWF temperatures at 00, 03, ..., 21 UTC are used.
    """
    if value_col == "temp_f":
        f = f[(f.model != "ecmwf_ifs025") | (f.valid_utc.dt.hour % 3 == 0)]
    return f


def _forecasts(filename: str, value_col: str) -> pd.DataFrame:
    f = usable_forecasts(pd.read_parquet(DATA / filename), value_col)
    f["available_at"] = f.valid_utc - pd.to_timedelta(f.lead_day, unit="D") + FORECAST_LATENCY
    return f


def _asos() -> pd.DataFrame:
    a = pd.read_parquet(DATA / "asos.parquet")
    return a.assign(available_at=a.valid_utc)


def _afd() -> pd.DataFrame:
    a = pd.read_parquet(DATA / "afd.parquet")
    a["issued_utc"] = pd.to_datetime(a.issued_utc, utc=True)
    return a.assign(available_at=a.issued_utc)


def _mos() -> pd.DataFrame:
    m = pd.read_parquet(DATA / "mos.parquet")
    m["runtime"] = pd.to_datetime(m.runtime, utc=True)
    m["ftime"] = pd.to_datetime(m.ftime, utc=True)
    m = m[m.runtime.dt.hour == 0]                       # SPEC 9: only the 00Z run of day D
    return m.assign(available_at=m.runtime + MOS_LATENCY)


@lru_cache(maxsize=1)
def sources() -> list[Source]:
    return [Source("daily", _daily, "NOAA ACIS daily high and precipitation", key="sid"),
            Source("fc_temp", lambda: _forecasts("forecasts.parquet", "temp_f"), "model hourly temperature", key="sid"),
            Source("fc_precip", lambda: _forecasts("precip_forecasts.parquet", "precip_in"),
                   "model hourly precipitation", key="sid"),
            Source("asos", _asos, "hourly station observations", key="sid"),
            Source("afd", _afd, "NWS Area Forecast Discussions", key="office"),
            Source("mos", _mos, "NWS GFS MOS guidance (00Z runs)", key="sid")]


# ---------------------------------------------------------------- features (all through the snapshot)
def local_hours_utc(sid: str, d: date, hours) -> pd.DatetimeIndex:
    tz = ZoneInfo(BY_SID[sid].tz)
    return pd.DatetimeIndex([pd.Timestamp(datetime.combine(d, time(h), tzinfo=tz)).tz_convert("UTC") for h in hours])


def model_highs(snap: Snapshot, sid: str, d: date) -> dict[str, float]:
    """Forecast high per model for day d: max of previous_day1 hourly values at local hours 00-20."""
    f = snap.get("fc_temp", key=sid)
    f = f[(f.lead_day == 1) & f.valid_utc.isin(local_hours_utc(sid, d, TEMP_HOURS))]
    # need most of the day: 15 of 21 hourly values (GFS, ICON) or 6 of 7 three-hourly values (ECMWF)
    return {m: round(float(g.temp_f.max()), 1) for m, g in f.groupby("model")
            if len(g) >= (6 if m == "ecmwf_ifs025" else 15)}


def model_precip_totals(snap: Snapshot, sid: str, d: date) -> dict[str, float]:
    """Precipitation per model over the climate day: previous_day1 for LST hours 00-20, previous_day2 for 21-23."""
    start = climate_day_start_utc(sid, d)
    f = snap.get("fc_precip", key=sid)
    hour = (f.valid_utc - start) / pd.Timedelta(hours=1)
    # hourly precipitation is the amount in the hour ending at valid time, so hour 1..24 covers the climate day
    use = ((hour >= 1) & (hour <= 21) & (f.lead_day == 1)) | ((hour >= 22) & (hour <= 24) & (f.lead_day == 2))
    f = f[use]
    return {m: round(float(g.precip_in.sum()), 3) for m, g in f.groupby("model") if len(g) >= 20}


def overnight_obs(snap: Snapshot, sid: str, d: date) -> dict:
    """Station observations from the start of the climate day to t."""
    start = climate_day_start_utc(sid, d)
    a = snap.get("asos", key=sid)
    a = a[a.valid_utc > start]
    return {"max_temp_f": None if a.tmpf.isna().all() else float(a.tmpf.max()),
            "min_temp_f": None if a.tmpf.isna().all() else float(a.tmpf.min()),
            "latest_temp_f": None if a.tmpf.dropna().empty else float(a.tmpf.dropna().iloc[-1]),
            "precip_in": round(float(a.p01i.clip(lower=0).sum()), 3), "reports": int(len(a))}


def recent_days(snap: Snapshot, sid: str, n: int = 7) -> pd.DataFrame:
    return snap.get("daily", key=sid).tail(n)[["date", "maxt", "pcpn"]]


def model_errors(snap: Snapshot, sid: str, d: date, n: int = 7) -> pd.DataFrame:
    """Forecast high minus observed high per model, for the n most recent days already observed at t."""
    rows = []
    for _, r in recent_days(snap, sid, n).iterrows():
        if pd.isna(r.maxt):
            continue
        for m, hi in model_highs(snap, sid, r.date).items():
            rows.append({"date": r.date, "model": m, "forecast_high": hi, "observed_high": r.maxt,
                         "error": round(hi - r.maxt, 1)})
    return pd.DataFrame(rows)


def mos_guidance(snap: Snapshot, sid: str, d: date) -> dict:
    """00Z GFS MOS run of day d: daytime max for d, and 6-hour precipitation chances overlapping its climate day."""
    m = snap.get("mos", key=sid)
    run = m[m.runtime == pd.Timestamp(datetime.combine(d, time(0))).tz_localize("UTC")]
    if run.empty:
        return {}
    start, end = climate_day_start_utc(sid, d), climate_day_start_utc(sid, d + timedelta(days=1))
    nx = run[run.ftime == pd.Timestamp(datetime.combine(d + timedelta(days=1), time(0))).tz_localize("UTC")].n_x
    p06 = run[(run.ftime > start) & (run.ftime - pd.Timedelta(hours=6) < end)].dropna(subset=["p06"])
    return {"max_temp_f": None if nx.dropna().empty else float(nx.dropna().iloc[0]),
            "p06": {t.strftime("%Y-%m-%dT%H:%MZ"): float(p) for t, p in zip(p06.ftime, p06.p06)}}


def latest_discussions(snap: Snapshot, office: str, n: int = 1) -> pd.DataFrame:
    return snap.get("afd", key=office).tail(n)


def ssp_temp_features(snap: Snapshot, q: dict) -> dict:
    """Features of the temperature SSP baseline (SPEC 5, renamed in 10.3). NaN where unavailable."""
    sid, d, thr = q["sid"], q["date"], q["threshold_f"]
    highs = model_highs(snap, sid, d)
    vals = list(highs.values())
    ob = overnight_obs(snap, sid, d)
    err = model_errors(snap, sid, d, n=1)
    yday = err.groupby("date").error.mean() if len(err) else pd.Series(dtype=float)
    doy = d.timetuple().tm_yday
    return {"n_models": len(vals),
            "mean_minus_thr": float(np.mean(vals)) - thr if vals else math.nan,
            "spread": float(np.max(vals) - np.min(vals)) if vals else math.nan,
            "overnight_max_minus_thr": (ob["max_temp_f"] - thr) if ob["max_temp_f"] is not None else math.nan,
            "yesterday_error": float(yday.iloc[-1]) if len(yday) else math.nan,
            "doy_sin": math.sin(2 * math.pi * doy / 365.25), "doy_cos": math.cos(2 * math.pi * doy / 365.25)}


def ssp_precip_features(snap: Snapshot, q: dict) -> dict:
    sid, d = q["sid"], q["date"]
    tot = model_precip_totals(snap, sid, d)
    doy = d.timetuple().tm_yday
    feats = {f"log1p_{m}": math.log1p(tot[m]) if m in tot else math.nan for m in MODELS}
    feats.update({"n_models": len(tot),
                  "n_ge_001": sum(v >= MEASURABLE_IN for v in tot.values()),
                  "n_ge_010": sum(v >= 0.10 for v in tot.values()),
                  "doy_sin": math.sin(2 * math.pi * doy / 365.25), "doy_cos": math.cos(2 * math.pi * doy / 365.25)})
    return feats
