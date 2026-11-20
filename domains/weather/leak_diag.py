"""Diagnose leak-check flags from the cached exact runs (no network: every request is already cached)."""
import pandas as pd

from domains.weather.build_benchmark import decision_time_utc
from domains.weather.checks import CACHE, CHECK_DAYS, CHECK_STATIONS, MODELS, TOL, fetch_runs, used_hours
from domains.weather.download import DATA
from core.http import CachedClient

client = CachedClient(CACHE / "openmeteo_single", min_interval_s=0)
arch = pd.read_parquet(DATA / "forecasts.parquet")
rows = []
for sid in CHECK_STATIONS:
    a = arch[arch.sid == sid].set_index(["valid_utc", "model", "lead_day"]).temp_f
    for d in CHECK_DAYS:
        runs, t = fetch_runs(client, sid, d), decision_time_utc(sid, d)
        for valid, lead in used_hours(sid, d, "temp_f"):
            for m in MODELS:
                v = a.get((valid, m, lead))
                if v is None:
                    continue
                c = runs[(runs.valid_utc == valid) & (runs.model == m)].dropna(subset=["temp_f"])
                hit = sorted(c[(c.temp_f - v).abs() <= TOL["temp_f"]].run)
                nominal = valid - pd.Timedelta(hours=24)        # the run a "previous_day1" value should come from
                rows.append({"sid": sid, "date": d, "model": m, "valid": valid, "value": v, "n_runs": len(c),
                             "hits": [h.strftime("%d %HZ") for h in hit],
                             "any_safe": any((t - h) >= pd.Timedelta(hours=6) for h in hit),
                             "only_unsafe": bool(hit) and not any((t - h) >= pd.Timedelta(hours=6) for h in hit),
                             "hit_lag_h": [round((valid - h) / pd.Timedelta(hours=1)) for h in hit],
                             "valid_hour_utc": valid.hour, "matched": bool(hit), "nominal": nominal})
r = pd.DataFrame(rows)
print("values:", len(r), "| matched:", round(r.matched.mean(), 3))
print("values whose ONLY matching runs are unsafe:", int(r.only_unsafe.sum()))
print(r[r.only_unsafe][["sid", "date", "model", "valid", "value", "hits", "hit_lag_h"]].to_string(index=False))
lags = r[r.matched].hit_lag_h.explode().astype(int)
print("valid-minus-init lag of matching runs (h):", lags.value_counts().sort_index().to_dict())
e = r[r.model == "ecmwf_ifs025"]
print("ECMWF matched share by valid hour UTC:", e.groupby("valid_hour_utc").matched.mean().round(2).to_dict())
print("ECMWF runs available per value:", e.n_runs.value_counts().to_dict())
u = r[r.hits.map(len) == 1].copy()
u["lag"] = u.hit_lag_h.map(lambda x: x[0])
print("UNIQUE matches:", len(u), "| lag min:", u.lag.min(), "| share lag >= 24h:", round((u.lag >= 24).mean(), 4))
print("unique lag < 24h by model / 3-hourly:", u[u.lag < 24].groupby(["model", u.valid_hour_utc % 3 == 0]).size().to_dict())
e3 = r[(r.model == "ecmwf_ifs025") & (r.valid_hour_utc % 3 == 0)]
print("ECMWF at 3-hourly times: n", len(e3), "matched", round(e3.matched.mean(), 3), "any_safe", round(e3.any_safe.mean(), 3))
g = r[r.model != "ecmwf_ifs025"]
print("GFS+ICON: n", len(g), "matched", round(g.matched.mean(), 3), "any_safe among matched", round(g[g.matched].any_safe.mean(), 4))
