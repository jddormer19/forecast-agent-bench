"""The weather agent's tools (SPEC sections 5, 9, 10). Every tool reads only through the question's Snapshot.

Run as an MCP server (stdio):  python -m domains.weather.tools --track temp
"""
from __future__ import annotations

import argparse
from typing import Callable, Literal

import pandas as pd

from core.pit import ToolBox
from domains.weather import sources as S
from domains.weather.stations import BY_SID

DISCUSSION_CHARS = 9000   # an AFD is usually 3-8k characters; longer ones are cut with a marker

QUESTION_TEXT = {
    "temp": "Will the official high temperature at {city} ({sid}) on {date} be at least {threshold_f}°F, "
            "its 1991-2020 normal high for that date?",
    "precip": "Will measurable precipitation (at least 0.01 inch) be recorded at {city} ({sid}) on {date}?",
}


def build_toolbox(questions: pd.DataFrame, track: str, log_path=None,
                  ssp: Callable[[str], float] | None = None) -> ToolBox:
    """Tools for one track. `ssp(qid)` returns the SSP baseline's probability (registered only when given)."""
    tb = ToolBox(S.sources(), questions, log_path=log_path)

    @tb.tool
    def get_question(snap, q) -> dict:
        """The question, the station, and the decision time. Call this first."""
        st = BY_SID[q["sid"]]
        local_t = pd.Timestamp(q["t_utc"]).tz_convert(st.tz)
        return {"question": QUESTION_TEXT[track].format(city=st.city, sid=st.sid, date=q["date"],
                                                       threshold_f=q.get("threshold_f")),
                "station": st.sid, "city": st.city, "nws_office": st.office, "date": str(q["date"]),
                "decision_time_local": local_t.strftime("%Y-%m-%d %H:%M %Z"),
                "note": "All tools return only information available at the decision time. The official day runs "
                        "midnight to midnight local standard time."}

    @tb.tool
    def model_forecasts(snap, q) -> dict:
        """Forecasts for the day from three weather models (GFS, ECMWF, ICON), from runs about a day old:
        the forecast high (°F, max over 00:00-20:00 local) and total precipitation (inches, whole day)."""
        return {"high_temp_f": S.model_highs(snap, q["sid"], q["date"]),
                "precip_total_in": S.model_precip_totals(snap, q["sid"], q["date"])}

    @tb.tool
    def mos_guidance(snap, q) -> dict:
        """Official NWS GFS MOS statistical guidance from today's 00Z run: daytime max temperature (°F) and
        6-hour precipitation chances (%, keyed by the end of each 6-hour period, UTC)."""
        return S.mos_guidance(snap, q["sid"], q["date"]) or {"unavailable": "no 00Z MOS run for this day"}

    @tb.tool
    def observations_so_far(snap, q) -> dict:
        """Station observations since the official day began (midnight local standard time) until now:
        max/min/latest temperature (°F) and precipitation so far (inches)."""
        return S.overnight_obs(snap, q["sid"], q["date"])

    @tb.tool
    def recent_weather(snap, q, days: int = 5) -> dict:
        """Official daily high (°F) and precipitation (inches) for the last `days` complete days (max 14)."""
        d = S.recent_days(snap, q["sid"], max(1, min(int(days), 14)))
        return {"days": [{"date": str(r.date), "high_f": None if pd.isna(r.maxt) else r.maxt,
                          "precip_in": None if pd.isna(r.pcpn) else r.pcpn} for r in d.itertuples()]}

    @tb.tool
    def model_recent_errors(snap, q, days: int = 7) -> dict:
        """How each model's forecast high compared with the observed high over the last `days` days (max 14).
        error = forecast - observed (°F); a positive mean means the model has been running warm."""
        e = S.model_errors(snap, q["sid"], q["date"], max(1, min(int(days), 14)))
        if e.empty:
            return {"unavailable": "no recent verified forecasts"}
        return {"mean_error_by_model": e.groupby("model").error.mean().round(1).to_dict(),
                "days": e.assign(date=e.date.astype(str)).to_dict("records")}

    @tb.tool
    def forecast_discussion(snap, q, which: Literal["latest", "previous"] = "latest") -> dict:
        """The local NWS office's Area Forecast Discussion: forecasters' written reasoning about the coming
        days. `latest` is the most recent one issued before now; `previous` is the one before it."""
        office = BY_SID[q["sid"]].office
        d = S.latest_discussions(snap, office, 2)
        if d.empty or (which == "previous" and len(d) < 2):
            return {"unavailable": "no discussion available"}
        r = d.iloc[-1] if which == "latest" else d.iloc[0]
        text = r.text if len(r.text) <= DISCUSSION_CHARS else r.text[:DISCUSSION_CHARS] + "\n[...cut]"
        return {"office": office, "issued_utc": r.issued_utc.strftime("%Y-%m-%d %H:%MZ"), "text": text}

    if ssp is not None:
        @tb.tool
        def statistical_baseline(snap, q) -> dict:
            """Probability from a simple statistical model fitted on 2024-2025 data, using the three models'
            forecasts and recent observations. Well calibrated on past data; it cannot read the discussions."""
            p = float(ssp(q["qid"]))
            return {"unavailable": "no baseline probability for this day"} if p != p else {"probability": round(p, 3)}

    return tb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=list(QUESTION_TEXT), required=True)
    ap.add_argument("--tools", default="", help="comma-separated allowlist (default: all)")
    a = ap.parse_args()
    qs = pd.read_parquet(S.DATA / f"questions_{a.track}.parquet")
    allowed = [t for t in a.tools.split(",") if t] or None
    build_toolbox(qs, a.track, log_path=S.DATA.parent.parent / "data" / "runs" / "tool_calls.jsonl") \
        .mcp_server(f"weather-{a.track}", allowed).run()
