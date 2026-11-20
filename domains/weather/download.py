"""Download the raw weather data (SPEC sections 4 and 7). Run: python -m domains.weather.download [sources...]

Sources and the provider limits each client respects (from the data-rights audit, 2026-11-19):
  acis       NOAA RCC ACIS daily highs + 1991-2020 normals     one request at a time, 1 per station
  forecasts  Open-Meteo Previous Runs, GFS/ECMWF/ICON hourly    1 s apart (limit 600/min, 10,000/day)
  afd        IEM AFOS archive, NWS Area Forecast Discussions    1 s apart (IEM throttle: 1/s per IP)
  asos       IEM ASOS hourly temperature + precipitation       1 s apart
  precip_forecasts  Open-Meteo Previous Runs, hourly precipitation   1 s apart (precipitation track, SPEC 9)
  mos        IEM archive of NWS GFS MOS guidance               1 s apart (precipitation track, SPEC 9)

IEM allows 1 request per second per public IP, and every machine on the author's network shares one IP:
never run IEM requests from two machines at once. Throttle pages ("Too many requests") are retried, never cached.

Everything is cached under .cache/ (gitignored); a re-run reads the cache and makes no network calls.
"""
from __future__ import annotations

import argparse
import calendar
import gzip
import io
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


def iem_throttled(text: str) -> bool:
    return text.lstrip()[:40].lower().startswith("too many requests")


def iem_client(name: str) -> CachedClient:
    return CachedClient(CACHE / name, min_interval_s=1.0, reject=iem_throttled)


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
                "elems": [{"name": "maxt"}, {"name": "maxt", "normal": "1"}, {"name": "pcpn"}]}
        data = json.loads(client.fetch("https://data.rcc-acis.org/StnData", json_body=body))["data"]
        rows += [(st.sid, *r) for r in data]
        print(f"acis {st.sid}: {len(data)} days")
    # raw strings ("M" missing, "T" trace, a trailing "A" flags a multi-day accumulation); parsed by the build step
    df = pd.DataFrame(rows, columns=["sid", "date", "maxt_raw", "normal_raw", "pcpn_raw"])
    _save(df, out / "acis.parquet")
    return df


# ---------- Open-Meteo previous runs ----------
def forecast_rows(sid: str, payload: dict, variable: str = "temperature_2m") -> list[tuple]:
    """Flatten one Previous Runs response into (sid, valid_utc, model, lead_day, value), skipping nulls."""
    h = payload["hourly"]
    out = []
    for lead in LEADS:
        for model in MODELS:
            vals = h.get(f"{variable}_previous_day{lead}_{model}")
            if vals is None:
                continue
            out += [(sid, t, model, lead, v) for t, v in zip(h["time"], vals) if v is not None]
    return out


def _previous_runs(variable: str, unit_params: dict, value_col: str, filename: str, out: Path) -> pd.DataFrame:
    client = CachedClient(CACHE / "openmeteo", min_interval_s=1.0)
    hourly = ",".join(f"{variable}_previous_day{lead}" for lead in LEADS)  # never the plain variable: it leaks
    rows = []
    for st in STATIONS:
        for a, b in months(START, END):
            # key order kept as in the first temperature download so its cache entries still match
            params = {"latitude": st.lat, "longitude": st.lon, "start_date": a.isoformat(), "end_date": b.isoformat(),
                      "hourly": hourly, "models": ",".join(MODELS), **unit_params, "timezone": "UTC"}
            payload = json.loads(client.fetch("https://previous-runs-api.open-meteo.com/v1/forecast", params))
            if payload.get("error"):
                raise RuntimeError(f"Open-Meteo {st.sid} {a}: {payload.get('reason')}")
            rows += forecast_rows(st.sid, payload, variable)
        print(f"{filename} {st.sid}: {len(rows)} rows so far ({client.network_calls} network calls)")
    df = pd.DataFrame(rows, columns=["sid", "valid_utc", "model", "lead_day", value_col])
    df["valid_utc"] = pd.to_datetime(df.valid_utc, utc=True)
    _save(df, out / f"{filename}.parquet")
    return df


def download_forecasts(out: Path = DATA) -> pd.DataFrame:
    return _previous_runs("temperature_2m", {"temperature_unit": "fahrenheit"}, "temp_f", "forecasts", out)


def download_precip_forecasts(out: Path = DATA) -> pd.DataFrame:
    return _previous_runs("precipitation", {"precipitation_unit": "inch"}, "precip_in", "precip_forecasts", out)


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
    client = iem_client("iem_afos")
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
    client = iem_client("iem_asos")
    frames = []
    for st in STATIONS:
        for year in range(START.year, END.year + 1):
            a, b = max(date(year, 1, 1), START), min(date(year + 1, 1, 1), END + timedelta(days=1))
            params = [("station", st.sid), ("data", "tmpf"), ("data", "p01i"), ("tz", "Etc/UTC"), ("format", "onlycomma"),
                      ("latlon", "no"), ("missing", "M"), ("report_type", "3"), ("report_type", "4"),
                      ("year1", a.year), ("month1", a.month), ("day1", a.day),
                      ("year2", b.year), ("month2", b.month), ("day2", b.day)]
            text = client.fetch("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params)
            frames.append(pd.read_csv(io.StringIO(text), dtype=str).rename(columns={"valid": "valid_utc"})
                          .assign(sid=st.sid)[["sid", "valid_utc", "tmpf", "p01i"]])
        print(f"asos {st.sid}: done ({client.network_calls} calls)")
    df = pd.concat(frames, ignore_index=True)
    df["valid_utc"] = pd.to_datetime(df.valid_utc, utc=True)
    df["tmpf"] = pd.to_numeric(df.tmpf, errors="coerce")  # "M" (missing) -> NaN
    # p01i: "T" (trace) becomes 0.0001 so it never counts as measurable (>= 0.01 in); "M" -> NaN
    df["p01i"] = pd.to_numeric(df.p01i.replace("T", "0.0001"), errors="coerce")
    _save(df.drop_duplicates(), out / "asos.parquet")
    return df


# ---------- IEM MOS archive (precipitation track, SPEC 9) ----------
def download_mos(out: Path = DATA) -> pd.DataFrame:
    client = iem_client("iem_mos")
    frames = []
    for st in STATIONS:
        for a, b in months(START, END):
            params = {"station": f"K{st.sid}", "model": "GFS", "format": "csv",
                      "sts": f"{a.isoformat()}T00:00Z", "ets": f"{(b + timedelta(days=1)).isoformat()}T00:00Z"}
            df = pd.read_csv(io.StringIO(client.fetch("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py", params)))
            if len(df):
                frames.append(df[["runtime", "ftime", "n_x", "p06", "p12"]].assign(sid=st.sid))
        print(f"mos {st.sid}: done ({client.network_calls} calls)")
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    _save(df, out / "mos.parquet")
    return df


def verify_cache() -> int:
    """Delete any cached IEM response that is a throttle page (expected: 0), so the next run re-fetches it."""
    bad = [p for p in CACHE.glob("iem_*/*.gz")
           if iem_throttled(gzip.decompress(p.read_bytes()).decode("utf-8", "replace"))]
    for p in bad:
        p.unlink()
    print(f"verify: {len(bad)} throttle pages removed from cache")
    return len(bad)


def _save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {path.relative_to(ROOT)}: {len(df):,} rows")


SOURCES = {"acis": download_acis, "forecasts": download_forecasts, "afd": download_afd, "asos": download_asos,
           "precip_forecasts": download_precip_forecasts, "mos": download_mos, "verify": verify_cache}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help=f"any of {', '.join(SOURCES)} (default: all)")
    names = ap.parse_args().sources or list(SOURCES)
    unknown = set(names) - set(SOURCES)
    if unknown:
        ap.error(f"unknown source(s): {', '.join(sorted(unknown))}")
    for name in names:
        SOURCES[name]()
