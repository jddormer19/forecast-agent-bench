"""SPEC 8.2 and 9: spot-check 20 temperature and 20 precipitation outcomes against the NWS daily climate reports.

Done 2026-11-22, AFTER the one-time holdout scoring (the check was pre-registered for before the agent runs and was
missed; see RESULTS.md "Pre-registered checks"). It cannot change any result: it only reports agreement.

For 2 primary-holdout questions per station and track (seed 2026), fetches the station's final CLI report for the
climate day (issued the next morning, Iowa Environmental Mesonet AFOS archive, at most 1 request per 1.5 s, cached),
and compares its observed maximum and precipitation with the ACIS values behind each outcome.

Run: python -m domains.weather.spot_check   ->  domains/weather/spot_check_results.json
"""
import json
import re
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

from domains.weather.build_benchmark import DATA, parse_int, parse_pcpn

OUT = Path(__file__).with_name("spot_check_results.json")
CACHE = DATA / "checks" / "cli"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
UA = {"User-Agent": "forecast-agent-bench/0.1"}
MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER",
          "NOVEMBER", "DECEMBER"]


def fetch(sid: str, d) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"CLI{sid}_{d}.txt"
    if f.exists():
        return f.read_text()
    nxt = d + timedelta(days=1)
    r = requests.get(URL, params={"pil": f"CLI{sid}", "sdate": str(d), "edate": str(nxt + timedelta(days=1)),
                                  "fmt": "text", "limit": 9999}, headers=UA, timeout=60)
    r.raise_for_status()
    time.sleep(1.5)
    f.write_text(r.text)
    return r.text


def final_summary(text: str, d) -> str | None:
    """The CLI issued after the climate day ended (reports it as YESTERDAY), not the same-day preliminary one."""
    tag = f"CLIMATE SUMMARY FOR {MONTHS[d.month - 1]} {d.day} {d.year}"
    for block in re.split(r"\x01|\n\$\$", text):
        if tag in block and "VALID TODAY AS OF" not in block and "YESTERDAY" in block:
            return block
    return None


def parse(block: str) -> tuple[int | None, float | None]:
    maxt = pcpn = None
    m = re.search(r"YESTERDAY\s*\n\s*MAXIMUM\s+(-?\d+|MM)", block)
    if m and m.group(1) != "MM":
        maxt = int(m.group(1))
    p = re.search(r"PRECIPITATION \(IN\)\s*\n\s*YESTERDAY\s+(\d+\.\d+|T|MM)", block)
    if p and p.group(1) != "MM":
        pcpn = 0.0 if p.group(1) == "T" else float(p.group(1))
    return maxt, pcpn


def sample(track: str) -> pd.DataFrame:
    b = pd.read_parquet(DATA / f"baselines_{track}.parquet")
    b = b[b.period == "holdout_primary"]
    return b.groupby("sid").sample(2, random_state=2026).sort_values(["sid", "date"]).reset_index(drop=True)


def main():
    acis = pd.read_parquet(DATA / "acis.parquet")
    acis["date"] = pd.to_datetime(acis.date).dt.date
    acis = acis.set_index(["sid", "date"])
    rows = []
    for track in ["temp", "precip"]:
        for q in sample(track).itertuples():
            d = pd.Timestamp(q.date).date()
            block = final_summary(fetch(q.sid, d), d)
            cli_max, cli_p = parse(block) if block else (None, None)
            a = acis.loc[(q.sid, d)]
            acis_max, acis_p = parse_int(a.maxt_raw), parse_pcpn(a.pcpn_raw)
            row = {"track": track, "sid": q.sid, "date": str(d), "outcome": int(q.outcome), "found": block is not None,
                   "cli_max": cli_max, "acis_max": acis_max, "cli_pcpn": cli_p, "acis_pcpn": acis_p}
            if track == "temp":
                row["value_match"] = cli_max == acis_max
                row["outcome_match"] = None if cli_max is None else int(cli_max >= q.threshold_f) == q.outcome
                row["threshold_f"] = float(q.threshold_f)
            else:
                row["value_match"] = None if cli_p is None or acis_p is None else abs(cli_p - acis_p) < 0.005
                row["outcome_match"] = None if cli_p is None else int(cli_p >= 0.01) == q.outcome
            rows.append(row)
            print(row)
    df = pd.DataFrame(rows)
    summary = {t: {"checked": int((g.track == t).sum()), "report_found": int(g.found.sum()),
                   "value_match": int(g.value_match.fillna(False).astype(bool).sum()),
                   "outcome_match": int(g.outcome_match.fillna(False).astype(bool).sum())}
               for t, g in df.groupby("track")}
    OUT.write_text(json.dumps({"note": "SPEC 8.2/9 spot-check, done 2026-11-22 after the one-time scoring; NWS CLI "
                                       "reports via the IEM AFOS archive. Cannot change any result.",
                               "summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
