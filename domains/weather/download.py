"""Download the raw weather data (SPEC sections 4 and 7). Run: python -m domains.weather.download [sources...]

Sources and the provider limits each client respects (from the data-rights audit, 2026-11-19):
  acis       NOAA RCC ACIS daily highs + 1991-2020 normals     one request at a time, 1 per station
  forecasts  Open-Meteo Previous Runs, GFS/ECMWF/ICON hourly    1 s apart (limit 600/min, 10,000/day)
  afd        IEM AFOS archive, NWS Area Forecast Discussions    1 s apart (IEM throttle: 1/s per IP)
  asos       IEM ASOS hourly station observations              1 s apart

Everything is cached under .cache/ (gitignored); a re-run reads the cache and makes no network calls.
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from core.http import CachedClient
from domains.weather.stations import STATIONS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "weather"
CACHE = ROOT / ".cache" / "weather"

# Benchmark dates are 2024-03-01..2026-11-15 (SPEC 3). Pad the start for "yesterday" features and overnight
# discussions, and the end by a day because west-coast local days run past midnight UTC.
START, END = date(2024, 2, 1), date(2026, 11, 16)
MODELS = ("gfs_global", "ecmwf_ifs025", "icon_global")   # pinned names (audit: avoid "seamless" blends)
LEADS = (1, 2)                                           # previous_day1 and previous_day2 (SPEC 4, 8)


def months(start: date, end: date):
    """Yield (first_day, last_day) for every calendar month overlapping [start, end]."""
    d = start.replace(day=1)
    while d <= end:
        last = d.replace(day=calendar.monthrange(d.year, d.month)[1])
        yield max(d, start), min(last, end)
        d = last + timedelta(days=1)


# ---------- ACIS ----------
def download_acis(out: Path = DATA) -> pd.DataFrame:
    client = CachedClient(CACHE / "acis", min_interval_s=1.0)
    rows = []
    for st in STATIONS:
        body = {"sid": st.sid, "sdate": START.isoformat(), "edate": END.isoformat(),
                "elems": [{"name": "maxt"}, {"name": "maxt", "normal": "1"}]}
        data = json.loads(client.fetch("https://data.rcc-acis.org/StnData", json_body=body))["data"]
        rows += [(st.sid, d, maxt, normal) for d, maxt, normal in data]
        print(f"acis {st.sid}: {len(data)} days")
    df = pd.DataFrame(rows, columns=["sid", "date", "maxt_raw", "normal_raw"])  # raw strings; flags parsed later
    _save(df, out / "acis.parquet")
    return df


# ---------- Open-Meteo previous runs ----------
def forecast_rows(sid: str, payload: dict) -> list[tuple]:
    """Flatten one Previous Runs response into (sid, valid_utc, model, lead_day, temp_f), skipping nulls."""
    h = payload["hourly"]
    out = []
    for lead in LEADS:
        for model in MODELS:
            vals = h.get(f"temperature_2m_previous_day{lead}_{model}")
            if vals is None:
                continue
            out += [(sid, t, model, lead, v) for t, v in zip(h["time"], vals) if v is not None]
    return out


def download_forecasts(out: Path = DATA) -> pd.DataFrame:
    client = CachedClient(CACHE / "openmeteo", min_interval_s=1.0)
    hourly = ",".join(f"temperature_2m_previous_day{lead}" for lead in LEADS)  # never plain temperature_2m: leaks
    rows = []
    for st in STATIONS:
        for a, b in months(START, END):
            params = {"latitude": st.lat, "longitude": st.lon, "start_date": a.isoformat(), "end_date": b.isoformat(),
                      "hourly": hourly, "models": ",".join(MODELS), "temperature_unit": "fahrenheit",
                      "timezone": "UTC"}
            payload = json.loads(client.fetch("https://previous-runs-api.open-meteo.com/v1/forecast", params))
            if payload.get("error"):
                raise RuntimeError(f"Open-Meteo {st.sid} {a}: {payload.get('reason')}")
            rows += forecast_rows(st.sid, payload)
        print(f"forecasts {st.sid}: {len(rows)} rows so far ({client.network_calls} network calls)")
    df = pd.DataFrame(rows, columns=["sid", "valid_utc", "model", "lead_day", "temp_f"])
    df["valid_utc"] = pd.to_datetime(df.valid_utc, utc=True)
    _save(df, out / "forecasts.parquet")
    return df


# ---------- IEM AFOS: Area Forecast Discussions ----------
HEADER = re.compile(r"\b([A-Z]{4}\d{2}) ([A-Z]{4}) (\d{2})(\d{2})(\d{2})\b")


def parse_afd_bulk(text: str, window_start: datetime, window_end: datetime) -> list[dict]:
    """Split a retrieve.py text dump into products and give each its full UTC issue time.

    The WMO header (e.g. "FXUS64 KFWD 142307") has only day/hour/minute. The archive lists products newest
    first, so walking down the list, each product's time is the latest calendar date that fits its
    day/hour/minute, lies inside the window (less a day of slack, since headers can precede archive entry
    by minutes), and is not later than the product listed above it.
    """
    floor = window_start - timedelta(days=1)
    month_starts, m = [], datetime(floor.year, floor.month, 1, tzinfo=timezone.utc)
    while m < window_end:
        month_starts.append(m)
        m = (m + timedelta(days=32)).replace(day=1)
    ceiling, out = window_end, []
    for chunk in text.split("\x01"):
        body = chunk.replace("\x03", "").strip()
        hit = HEADER.search(body[:300])
        if not hit:
            continue
        day, hh, mm = int(hit[3]), int(hit[4]), int(hit[5])
        candidates = []
        for ms in month_starts:
            if day <= calendar.monthrange(ms.year, ms.month)[1]:
                t = ms.replace(day=day, hour=hh, minute=mm)
                if floor <= t <= ceiling:
                    candidates.append(t)
        if not candidates:
            continue
        issued = max(candidates)
        ceiling = issued
        out.append({"issued_utc": issued, "wmo": f"{hit[1]} {hit[2]} {hit[3]}{hit[4]}{hit[5]}", "text": body})
    return out


def download_afd(out: Path = DATA) -> pd.DataFrame:
    client = CachedClient(CACHE / "iem_afos", min_interval_s=1.0)
    rows = []
    for office in sorted({s.office for s in STATIONS}):
        for a, b in months(START, END):
            ws = datetime(a.year, a.month, a.day, tzinfo=timezone.utc) - timedelta(days=1)  # overlap 1 day
            we = datetime(b.year, b.month, b.day, tzinfo=timezone.utc) + timedelta(days=1)  # edate is exclusive
            params = {"pil": f"AFD{office}", "sdate": ws.date().isoformat(), "edate": we.date().isoformat(),
                      "limit": 9999, "fmt": "text"}
            text = client.fetch("https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py", params)
            rows += [{"office": office, **p} for p in parse_afd_bulk(text, ws, we)]
        print(f"afd {office}: {sum(r['office'] == office for r in rows)} products ({client.network_calls} calls)")
    df = pd.DataFrame(rows).drop_duplicates(["office", "issued_utc", "wmo", "text"]).sort_values(["office", "issued_utc"])
    _save(df, out / "afd.parquet")
    return df


# ---------- IEM ASOS hourly observations ----------
def download_asos(out: Path = DATA) -> pd.DataFrame:
    client = CachedClient(CACHE / "iem_asos", min_interval_s=1.0)
    frames = []
    for st in STATIONS:
        for year in range(START.year, END.year + 1):
            a, b = max(date(year, 1, 1), START), min(date(year + 1, 1, 1), END + timedelta(days=1))
            params = [("station", st.sid), ("data", "tmpf"), ("tz", "Etc/UTC"), ("format", "onlycomma"),
                      ("latlon", "no"), ("missing", "M"), ("report_type", "3"), ("report_type", "4"),
                      ("year1", a.year), ("month1", a.month), ("day1", a.day),
                      ("year2", b.year), ("month2", b.month), ("day2", b.day)]
            text = client.fetch("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params)
            lines = [ln.split(",") for ln in text.strip().splitlines()[1:]]
            frames.append(pd.DataFrame([(st.sid, v, t) for _, v, t in lines if t != "M"],
                                       columns=["sid", "valid_utc", "tmpf"]))
        print(f"asos {st.sid}: done ({client.network_calls} calls)")
    df = pd.concat(frames, ignore_index=True)
    df["valid_utc"] = pd.to_datetime(df.valid_utc, utc=True)
    df["tmpf"] = df.tmpf.astype(float)
    _save(df.drop_duplicates(), out / "asos.parquet")
    return df


def _save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {path.relative_to(ROOT)}: {len(df):,} rows")


SOURCES = {"acis": download_acis, "forecasts": download_forecasts, "afd": download_afd, "asos": download_asos}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help=f"any of {', '.join(SOURCES)} (default: all)")
    names = ap.parse_args().sources or list(SOURCES)
    unknown = set(names) - set(SOURCES)
    if unknown:
        ap.error(f"unknown source(s): {', '.join(sorted(unknown))}")
    for name in names:
        SOURCES[name]()
