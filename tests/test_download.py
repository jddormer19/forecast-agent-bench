from datetime import datetime, timezone

import pytest
import requests

from core.http import CachedClient
from domains.weather.download import forecast_rows, months, parse_afd_bulk


class FakeResponse:
    def __init__(self, status, text="", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), 0, {}

    def request(self, *a, **k):
        self.calls += 1
        return self.responses.pop(0)


def client(tmp_path, responses, interval=1.0):
    clock = {"t": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s); clock["t"] += s

    c = CachedClient(tmp_path, interval, session=FakeSession(responses), sleep=sleep, clock=lambda: clock["t"])
    return c, sleeps


def test_cache_means_one_network_call(tmp_path):
    c, _ = client(tmp_path, [FakeResponse(200, "hello")])
    assert c.fetch("https://x", {"a": 1}) == "hello"
    assert c.fetch("https://x", {"a": 1}) == "hello"
    assert c.session.calls == 1


def test_retries_429_then_succeeds(tmp_path):
    c, sleeps = client(tmp_path, [FakeResponse(429), FakeResponse(200, "ok")])
    assert c.fetch("https://x") == "ok"
    assert c.session.calls == 2 and max(sleeps) >= 2.0


def test_throttle_spaces_requests(tmp_path):
    c, sleeps = client(tmp_path, [FakeResponse(200, "a"), FakeResponse(200, "b")], interval=1.0)
    c.fetch("https://x/1"); c.fetch("https://x/2")
    assert sleeps == [1.0]


def test_gives_up_after_retries(tmp_path):
    c, _ = client(tmp_path, [FakeResponse(503)] * 6)
    c.max_retries = 5
    with pytest.raises(requests.HTTPError):
        c.fetch("https://x")


def test_months_cover_range_exactly():
    from datetime import date
    m = list(months(date(2024, 2, 10), date(2024, 4, 5)))
    assert m == [(date(2024, 2, 10), date(2024, 2, 29)), (date(2024, 3, 1), date(2024, 3, 31)),
                 (date(2024, 4, 1), date(2024, 4, 5))]


def product(header, words="body"):
    return f"\x01\n123 \n{header}\nAFDXXX\n\nArea Forecast Discussion\n{words}\n\x03"


def test_afd_dates_resolved_across_month_boundary():
    utc = timezone.utc
    ws, we = datetime(2023, 12, 31, tzinfo=utc), datetime(2024, 2, 2, tzinfo=utc)
    # newest first, as the archive returns them: Feb 1, Jan 31, Jan 1, Dec 31
    text = "".join(product(h) for h in ["FXUS64 KFWD 010900", "FXUS64 KFWD 312200",
                                        "FXUS64 KFWD 011000", "FXUS64 KFWD 312300"])
    got = [p["issued_utc"] for p in parse_afd_bulk(text, ws, we)]
    assert got == [datetime(2024, 2, 1, 9, 0, tzinfo=utc), datetime(2024, 1, 31, 22, 0, tzinfo=utc),
                   datetime(2024, 1, 1, 10, 0, tzinfo=utc), datetime(2023, 12, 31, 23, 0, tzinfo=utc)]


def test_afd_skips_chunks_without_header():
    utc = timezone.utc
    got = parse_afd_bulk("\x01junk\x03" + product("FXUS64 KFWD 142307"),
                         datetime(2024, 6, 30, tzinfo=utc), datetime(2024, 8, 2, tzinfo=utc))
    assert len(got) == 1 and got[0]["issued_utc"] == datetime(2024, 7, 14, 23, 7, tzinfo=utc)


def test_forecast_rows_skip_nulls_and_keep_lead():
    payload = {"hourly": {"time": ["2024-07-15T00:00", "2024-07-15T01:00"],
                          "temperature_2m_previous_day1_gfs_global": [90.0, None],
                          "temperature_2m_previous_day2_ecmwf_ifs025": [88.5, 87.0]}}
    rows = forecast_rows("DFW", payload)
    assert ("DFW", "2024-07-15T00:00", "gfs_global", 1, 90.0) in rows
    assert len(rows) == 3
