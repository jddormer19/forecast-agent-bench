from datetime import date

import pandas as pd

from domains.weather.build_benchmark import (build, climate_day_start_utc, decision_time_utc, parse_int,
                                             parse_pcpn)


def test_parsers():
    assert parse_int("75") == 75 and parse_int("-3") == -3
    assert parse_int("M") is None and parse_int("75A") is None
    assert parse_pcpn("T") == 0.0 and parse_pcpn("0.12") == 0.12
    assert parse_pcpn("M") is None and parse_pcpn("1.20A") is None


def test_decision_time_follows_daylight_saving():
    assert str(decision_time_utc("DFW", date(2025, 1, 15))) == "2025-01-15 12:00:00+00:00"   # CST
    assert str(decision_time_utc("DFW", date(2025, 7, 15))) == "2025-07-15 11:00:00+00:00"   # CDT
    assert str(decision_time_utc("PHX", date(2025, 7, 15))) == "2025-07-15 13:00:00+00:00"   # no DST


def test_climate_day_uses_standard_time_all_year():
    assert str(climate_day_start_utc("DFW", date(2025, 7, 15))) == "2025-07-15 06:00:00+00:00"
    assert str(climate_day_start_utc("NYC", date(2025, 1, 15))) == "2025-01-15 05:00:00+00:00"


def acis_rows(rows):
    return pd.DataFrame(rows, columns=["sid", "date", "maxt_raw", "normal_raw", "pcpn_raw"])


def test_build_outcomes_splits_and_exclusions():
    acis = acis_rows([
        ("DFW", "2025-03-01", "70", "68", "0.00"),   # temp yes, dry
        ("DFW", "2025-03-02", "60", "68", "T"),      # temp no, trace = dry
        ("DFW", "2025-03-03", "M", "68", "0.30"),    # temp excluded, wet but decided before t (ASOS)
        ("DFW", "2026-03-01", "68", "68", "0.25A"),  # holdout; equal counts as yes; accumulated -> excluded
        ("DFW", "2023-12-31", "50", "55", "0.00"),   # outside range
    ])
    asos = pd.DataFrame({"sid": ["DFW", "DFW"],
                         "valid_utc": pd.to_datetime(["2025-03-03 08:53", "2025-03-03 13:53"], utc=True),
                         "tmpf": [50.0, 52.0], "p01i": [0.02, 0.30]})  # 0.02 by 06:00 CST (12Z); 13:53Z is after t
    temp, precip, m = build(acis, asos)
    assert temp.set_index("qid").outcome.to_dict() == {"DFW-2025-03-01": 1, "DFW-2025-03-02": 0, "DFW-2026-03-01": 1}
    assert temp.set_index("qid").split["DFW-2026-03-01"] == "holdout"
    assert m["temp"]["exclusions"]["missing_or_flagged_maxt"] == 1
    assert precip.set_index("qid").outcome.to_dict() == {"DFW-2025-03-01": 0, "DFW-2025-03-02": 0}
    assert m["precip"]["exclusions"] == {"missing_or_accumulated_pcpn": 1, "already_measurable_before_t": 1}


def test_fingerprint_is_stable_and_sensitive():
    acis = acis_rows([("DFW", "2025-03-01", "70", "68", "0.00"), ("DFW", "2025-03-02", "60", "68", "0.00")])
    _, _, m1 = build(acis, None)
    _, _, m2 = build(acis.iloc[::-1], None)
    acis.loc[0, "maxt_raw"] = "60"
    _, _, m3 = build(acis, None)
    assert m1["temp"]["fingerprint"] == m2["temp"]["fingerprint"] != m3["temp"]["fingerprint"]
