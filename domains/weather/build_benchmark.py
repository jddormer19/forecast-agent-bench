"""Build the two frozen question sets (SPEC sections 1, 3 and 9). Run: python -m domains.weather.build_benchmark

Writes data/weather/questions_{temp,precip}.parquet and benchmark_manifest.json (counts, exclusions,
fingerprints). Only questions, thresholds, outcomes and decision times are built here: no forecast is joined to
any outcome, and nothing is fitted. A later run must reproduce the same fingerprints (--check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from domains.weather.stations import BY_SID

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "weather"
FIRST, LAST = date(2024, 3, 1), date(2026, 11, 15)
HOLDOUT_START = date(2026, 1, 1)
DECISION_LOCAL = time(6, 0)
MEASURABLE_IN = 0.01


def parse_int(raw: str) -> int | None:
    """ACIS whole-degree value; None for missing ("M") or any flagged value (non-digit suffix)."""
    raw = str(raw).strip()
    return int(raw) if raw.lstrip("-").isdigit() else None


def parse_pcpn(raw: str) -> float | None:
    """ACIS precipitation: "T" (trace) -> 0.0; "M" or flagged (e.g. "0.45A", multi-day accumulation) -> None."""
    raw = str(raw).strip()
    if raw == "T":
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return None


def decision_time_utc(sid: str, d: date) -> pd.Timestamp:
    """06:00 local clock time (DST-aware) on day d, in UTC."""
    tz = ZoneInfo(BY_SID[sid].tz)
    return pd.Timestamp(datetime.combine(d, DECISION_LOCAL, tzinfo=tz)).tz_convert("UTC")


def climate_day_start_utc(sid: str, d: date) -> pd.Timestamp:
    """Start of the NWS climate day: midnight local STANDARD time (no DST shift), in UTC."""
    std_offset = datetime(d.year, 1, 15, tzinfo=ZoneInfo(BY_SID[sid].tz)).utcoffset()
    return pd.Timestamp(datetime.combine(d, time(0, 0))) .tz_localize("UTC") - std_offset


def fingerprint(df: pd.DataFrame, cols: list[str]) -> str:
    rows = df.sort_values("qid")[cols].astype(str).agg("|".join, axis=1)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


def precip_before_t(asos: pd.DataFrame) -> dict[tuple[str, date], float]:
    """Hourly precipitation accumulated from the climate-day start to t (inclusive), per (sid, date)."""
    out = {}
    for sid, g in asos.dropna(subset=["p01i"]).groupby("sid"):
        g = g.set_index("valid_utc").sort_index()
        for d in pd.date_range(FIRST, LAST).date:
            a, t = climate_day_start_utc(sid, d), decision_time_utc(sid, d)
            # p01i is the amount in the hour ending at the report, so a report after `a` covers time after `a`
            out[(sid, d)] = float(g.loc[(g.index > a) & (g.index <= t), "p01i"].clip(lower=0).sum())
    return out


def build(acis: pd.DataFrame, asos: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    a = acis.copy()
    a["date"] = pd.to_datetime(a.date).dt.date
    a = a[(a.date >= FIRST) & (a.date <= LAST)]
    a["maxt"] = a.maxt_raw.map(parse_int)
    a["normal"] = a.normal_raw.map(parse_int)
    a["qid"] = a.sid + "-" + a.date.astype(str)
    a["t_utc"] = [decision_time_utc(s, d) for s, d in zip(a.sid, a.date)]
    a["split"] = ["holdout" if d >= HOLDOUT_START else "dev" for d in a.date]
    manifest = {"range": [FIRST.isoformat(), LAST.isoformat()], "holdout_start": HOLDOUT_START.isoformat()}

    # temperature track (SPEC 1)
    ex_t = {"missing_or_flagged_maxt": int(a.maxt.isna().sum()), "missing_normal": int(a.normal.isna().sum())}
    temp = a.dropna(subset=["maxt", "normal"]).copy()
    temp["threshold_f"] = temp.normal.astype(int)
    temp["outcome"] = (temp.maxt >= temp.threshold_f).astype(int)
    temp = temp[["qid", "sid", "date", "t_utc", "split", "threshold_f", "outcome"]].sort_values("qid")
    manifest["temp"] = _summary(temp, ex_t, ["qid", "threshold_f", "outcome"])

    # precipitation track (SPEC 9)
    p = a.copy()
    p["pcpn"] = p.pcpn_raw.map(parse_pcpn) if "pcpn_raw" in p else None
    ex_p = {"missing_or_accumulated_pcpn": int(p.pcpn.isna().sum())}
    p = p.dropna(subset=["pcpn"])
    if asos is not None:
        before = precip_before_t(asos)
        p["precip_before_t_in"] = [before.get((s, d), 0.0) for s, d in zip(p.sid, p.date)]
        decided = p.precip_before_t_in >= MEASURABLE_IN
        ex_p["already_measurable_before_t"] = int(decided.sum())
        p = p[~decided]
    p["outcome"] = (p.pcpn >= MEASURABLE_IN).astype(int)
    precip = p[["qid", "sid", "date", "t_utc", "split", "outcome"]].sort_values("qid")
    manifest["precip"] = _summary(precip, ex_p, ["qid", "outcome"])
    return temp, precip, manifest


def _summary(df: pd.DataFrame, exclusions: dict, fp_cols: list[str]) -> dict:
    return {"questions": len(df), "dev": int((df.split == "dev").sum()), "holdout": int((df.split == "holdout").sum()),
            "per_station": df.groupby("sid").size().to_dict(), "exclusions": exclusions,
            "fingerprint": fingerprint(df, fp_cols)}
    # base rates are deliberately NOT computed here: outcomes are not summarized before the baselines are fitted


def main(check: bool):
    acis = pd.read_parquet(DATA / "acis.parquet")
    asos = pd.read_parquet(DATA / "asos.parquet") if (DATA / "asos.parquet").exists() else None
    if asos is not None and "p01i" not in asos:
        raise SystemExit("asos.parquet has no p01i column yet: re-run the asos download first (SPEC 9)")
    temp, precip, manifest = build(acis, asos)
    path = DATA / "benchmark_manifest.json"
    if check:
        old = json.loads(path.read_text())
        for track in ("temp", "precip"):
            assert old[track]["fingerprint"] == manifest[track]["fingerprint"], f"{track} fingerprint changed"
        print("fingerprints match:", {k: manifest[k]["fingerprint"] for k in ("temp", "precip")})
        return
    temp.to_parquet(DATA / "questions_temp.parquet", index=False)
    precip.to_parquet(DATA / "questions_precip.parquet", index=False)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: {kk: v[kk] for kk in ("questions", "dev", "holdout", "exclusions", "fingerprint")}
                      for k, v in manifest.items() if k in ("temp", "precip")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify fingerprints against the saved manifest")
    main(ap.parse_args().check)
