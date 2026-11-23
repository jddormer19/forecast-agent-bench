"""The per-source availability rules in domains/weather/sources.py: the real leak boundary (SPEC 4, 9, 10).

Each test writes a tiny synthetic parquet file into a temporary data folder, so no downloaded data is needed.
"""
import pandas as pd
import pytest

from core.pit import Snapshot, Source
from domains.weather import sources
from domains.weather.build_benchmark import decision_time_utc


def ts(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "DATA", tmp_path)
    return tmp_path


def visible(loader, t) -> pd.DataFrame:
    return Snapshot({"s": Source("s", loader)}, t).get("s")


def test_observed_high_is_hidden_until_its_climate_day_has_ended(data):
    pd.DataFrame({"sid": ["NYC"], "date": ["2026-07-15"], "maxt_raw": ["90"], "pcpn_raw": ["0.10"]}).to_parquet(data / "acis.parquet")
    d = sources._daily()
    # climate day = midnight local STANDARD time; Jul 16 00:00 EST = 05:00 UTC, plus the 1 h margin
    assert d.available_at.iloc[0] == ts("2026-07-16 06:00")
    t = decision_time_utc("NYC", pd.Timestamp("2026-07-15").date())          # 06:00 EDT on the question day
    assert visible(sources._daily, t).empty
    assert len(visible(sources._daily, ts("2026-07-16 06:00"))) == 1


def test_forecast_rule_and_ecmwf_temperature_only_at_3_hourly_times(data):
    pd.DataFrame({"model": ["gfs_global", "ecmwf_ifs025", "ecmwf_ifs025"],
                  "valid_utc": [ts("2026-07-15 16:00"), ts("2026-07-15 15:00"), ts("2026-07-15 16:00")],
                  "lead_day": [1, 1, 1], "temp_f": [88.0, 87.0, 87.5]}).to_parquet(data / "forecasts.parquet")
    f = sources._forecasts("forecasts.parquet", "temp_f")
    assert sorted(zip(f.model, f.valid_utc.dt.hour)) == [("ecmwf_ifs025", 15), ("gfs_global", 16)]   # interpolated ECMWF hour dropped
    # previous_day1: valid time - 1 day + 6 h latency
    assert f.loc[f.model == "gfs_global", "available_at"].iloc[0] == ts("2026-07-14 22:00")


def test_precipitation_keeps_every_ecmwf_hour(data):
    pd.DataFrame({"model": ["ecmwf_ifs025"], "valid_utc": [ts("2026-07-15 16:00")], "lead_day": [1],
                  "precip_in": [0.02]}).to_parquet(data / "precip_forecasts.parquet")
    assert len(sources._forecasts("precip_forecasts.parquet", "precip_in")) == 1


def test_mos_uses_only_the_00z_run_after_its_publication_latency(data):
    pd.DataFrame({"runtime": [ts("2026-07-15 00:00"), ts("2026-07-15 12:00")],
                  "ftime": [ts("2026-07-15 18:00"), ts("2026-07-16 00:00")], "station": ["KNYC", "KNYC"]}
                 ).to_parquet(data / "mos.parquet")
    m = sources._mos()
    assert m.runtime.dt.hour.tolist() == [0]
    assert m.available_at.iloc[0] == ts("2026-07-15 04:00")
    # the earliest decision time anywhere is 06:00 EDT = 10:00 UTC, so the 00Z run is always visible at t
    assert len(visible(sources._mos, decision_time_utc("NYC", pd.Timestamp("2026-07-15").date()))) == 1


def test_discussion_issued_after_t_is_invisible(data):
    pd.DataFrame({"issued_utc": ["2026-07-15T09:30:00Z", "2026-07-15T10:30:00Z"], "office": ["OKX", "OKX"],
                  "text": ["early", "late"]}).to_parquet(data / "afd.parquet")
    seen = visible(sources._afd, decision_time_utc("NYC", pd.Timestamp("2026-07-15").date()))
    assert seen.text.tolist() == ["early"]


def test_observations_are_available_at_their_report_time(data):
    pd.DataFrame({"valid_utc": [ts("2026-07-15 09:51"), ts("2026-07-15 10:51")], "sid": ["NYC", "NYC"],
                  "tmpf": [75.0, 77.0], "p01i": [0.0, 0.0]}).to_parquet(data / "asos.parquet")
    assert visible(sources._asos, ts("2026-07-15 10:00")).tmpf.tolist() == [75.0]
