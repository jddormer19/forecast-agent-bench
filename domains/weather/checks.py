"""Pre-run checks (SPEC section 8). Run: python -m domains.weather.checks leak

leak: which model run did each `previous_dayN` value we use actually come from? Where Open-Meteo's Single Runs
API has exact runs (Aug 2026 on), fetch every run initialized from 12Z two days before D through 06Z on D, match
each archived value to the run(s) that produced it, and compare that run's initialization with the decision time
t. A value is safe if its run was initialized at least SAFE_LATENCY before t (runs are published a few hours after
initialization). Any value whose run was initialized after t - SAFE_LATENCY fails the check.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from core.http import CachedClient
from domains.weather.build_benchmark import decision_time_utc
from domains.weather.download import CACHE, DATA, MODELS
from domains.weather.stations import BY_SID

CHECK_STATIONS = ("NYC", "ORD", "DEN", "PHX", "LAX")      # one or more per time zone
CHECK_DAYS = pd.date_range("2026-08-03", "2026-11-15", freq="7D").date
SAFE_LATENCY = timedelta(hours=6)
TOL = {"temp_f": 0.051, "precip_in": 0.0051}              # archived values are rounded to 0.1 °F / 0.01 in


def runs_for(d: date) -> list[datetime]:
    start = datetime.combine(d - timedelta(days=2), time(12), tzinfo=timezone.utc)
    return [start + timedelta(hours=6 * i) for i in range(8)]   # D-2 12Z ... D 06Z


def fetch_runs(client: CachedClient, sid: str, d: date) -> pd.DataFrame:
    st, frames = BY_SID[sid], []
    for run in runs_for(d):
        params = {"latitude": st.lat, "longitude": st.lon, "run": run.strftime("%Y-%m-%dT%H:%M"),
                  "hourly": "temperature_2m,precipitation", "models": ",".join(MODELS),
                  "temperature_unit": "fahrenheit", "precipitation_unit": "inch", "timezone": "UTC", "forecast_days": 4}
        h = json.loads(client.fetch("https://single-runs-api.open-meteo.com/v1/forecast", params)).get("hourly")
        if not h:
            continue
        for m in MODELS:
            frames.append(pd.DataFrame({"valid_utc": pd.to_datetime(h["time"], utc=True), "model": m, "run": run,
                                        "temp_f": h.get(f"temperature_2m_{m}"), "precip_in": h.get(f"precipitation_{m}")}))
    return pd.concat(frames, ignore_index=True)


def used_hours(sid: str, d: date, var: str) -> list[tuple[pd.Timestamp, int]]:
    """(valid_utc, lead_day) pairs the baselines and tools use for day d (SPEC 4 and 9)."""
    tz = ZoneInfo(BY_SID[sid].tz)
    if var == "temp_f":   # local clock hours 00:00-20:00, previous_day1
        return [(pd.Timestamp(datetime.combine(d, time(h), tzinfo=tz)).tz_convert("UTC"), 1) for h in range(21)]
    std = datetime(d.year, 1, 15, tzinfo=tz).utcoffset()   # precipitation: local standard hours
    base = pd.Timestamp(datetime.combine(d, time(0))).tz_localize("UTC") - std
    return [(base + pd.Timedelta(hours=h), 1 if h <= 20 else 2) for h in range(24)]


def leak_check() -> dict:
    client = CachedClient(CACHE / "openmeteo_single", min_interval_s=1.0)
    from domains.weather.sources import usable_forecasts   # the same filter the baselines and tools use
    archived = {"temp_f": usable_forecasts(pd.read_parquet(DATA / "forecasts.parquet"), "temp_f")}
    if (DATA / "precip_forecasts.parquet").exists():
        archived["precip_in"] = usable_forecasts(pd.read_parquet(DATA / "precip_forecasts.parquet"), "precip_in")
    rows = []
    for sid in CHECK_STATIONS:
        for d in CHECK_DAYS:
            runs = fetch_runs(client, sid, d)
            t = decision_time_utc(sid, d)
            for var, arch in archived.items():
                a = arch[arch.sid == sid].set_index(["valid_utc", "model", "lead_day"])[var]
                for valid, lead in used_hours(sid, d, var):
                    for m in MODELS:
                        v = a.get((valid, m, lead))
                        if v is None or pd.isna(v):
                            continue
                        cand = runs[(runs.valid_utc == valid) & (runs.model == m)].dropna(subset=[var])
                        hits = cand[(cand[var] - v).abs() <= TOL[var]].run
                        latest = hits.max() if len(hits) else None
                        safe = [(t - h) >= SAFE_LATENCY for h in hits]
                        rows.append({"var": var, "sid": sid, "date": d, "model": m, "lead": lead,
                                     "matched": latest is not None, "unique": len(hits) == 1,
                                     "any_safe": any(safe), "unique_unsafe": len(hits) == 1 and not safe[0],
                                     "hours_before_t": None if latest is None else (t - latest).total_seconds() / 3600})
        print(f"leak check {sid}: {client.network_calls} network calls so far")
    df = pd.DataFrame(rows)
    out = {}
    for (var, m), g in df.groupby(["var", "model"]):
        mt = g[g.matched]
        out[f"{var}/{m}"] = {"values": len(g), "matched": round(len(mt) / len(g), 3),
                             "unique_match": round(g.unique.mean(), 3),
                             "matched_with_a_safe_run": None if mt.empty else round(mt.any_safe.mean(), 4),
                             "unique_matches_unsafe": int(g.unique_unsafe.sum()),
                             # v1 criterion, kept for the record: credits the latest matching run, which flags
                             # repeated values (e.g. 0.00 in precipitation) that also match older runs
                             "v1_latest_match_unsafe_share": None if mt.empty else
                             round((mt.hours_before_t < SAFE_LATENCY.total_seconds() / 3600).mean(), 4)}
    # v2 criterion (SPEC change log 2026-11-19): every used value matches a run published >= 6 h before t, and no
    # value that matches exactly one run points to an unsafe run
    passed = all(v["matched"] >= 0.9 and v["matched_with_a_safe_run"] == 1.0 and v["unique_matches_unsafe"] == 0
                 for v in out.values())
    result = {"passed": passed, "safe_latency_hours": SAFE_LATENCY.total_seconds() / 3600,
              "stations": CHECK_STATIONS, "days": [str(d) for d in CHECK_DAYS], "by_variable_model": out}
    (DATA / "checks").mkdir(exist_ok=True)
    (DATA / "checks" / "leak_check.json").write_text(json.dumps(result, indent=2, default=str))
    df.to_parquet(DATA / "checks" / "leak_check_rows.parquet", index=False)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["leak"])
    {"leak": leak_check}[ap.parse_args().check]()
