# Weather benchmark SPEC (v1, FROZEN 2026-11-19)

Status: FROZEN on 2026-11-19, approved by the author before any data was downloaded or any model fitted. Any
later change is appended to the change log at the bottom with its date and reason, and no change may use holdout results.

## 1. Question

**Can an LLM agent that reads forecasters' written discussions beat classical statistical correction of
weather-model forecasts?**

Each benchmark item is one binary question:

> At 06:00 local time on day D: will the official daily high temperature at station S on day D be at or above
> its 1991-2020 normal high for that date?

- **Outcome:** NOAA station daily maximum (`maxt`, whole °F) from RCC ACIS, for the NWS climate day (midnight
  to midnight local *standard* time). `1` if `maxt >= threshold`, else `0`.
- **Threshold:** the ACIS 1991-2020 daily normal maximum for station S and date D (whole °F). It depends only
  on the calendar date, never on any forecast, so no question is selected using information about the outcome.
- **Decision time t:** 06:00 local clock time on day D (America/New_York, Chicago, Denver, Phoenix,
  Los_Angeles as applicable). Nothing published after t may reach any forecaster, baseline or agent.

## 2. Stations

| Station (ACIS id) | City | NWS office (AFD) |
| --- | --- | --- |
| NYC (Central Park) | New York | OKX |
| BOS (Logan) | Boston | BOX |
| ATL (Hartsfield-Jackson) | Atlanta | FFC |
| MIA (Miami Intl) | Miami | MFL |
| ORD (O'Hare) | Chicago | LOT |
| DFW (DFW Airport) | Dallas-Fort Worth | FWD |
| DEN (Denver Intl) | Denver | BOU |
| PHX (Sky Harbor) | Phoenix | PSR |
| LAX (LAX Airport) | Los Angeles | LOX |
| SEA (Sea-Tac) | Seattle | SEW |

Chosen for climate variety (marine, desert, mountain, subtropical, continental) and complete records.

## 3. Date range and split

- Range: 2024-03-01 (first date all three forecast models are archived) to 2026-11-15.
- **Dev:** 2024-03-01 to 2025-12-31 (about 6,700 questions). All development, fitting and prompt work.
- **Holdout:** 2026-01-01 to 2026-11-15 (about 3,190 questions). Scored once, after the code, prompts and
  baseline are frozen at a tagged commit. A second holdout run is refused by the scorer.
- Exclusions (counted and reported): missing or flagged `maxt`, missing normal, or fewer than 2 of the 3
  forecast models available at t.
- The benchmark file is frozen with a SHA-256 fingerprint of its question ids, thresholds and outcomes; the
  build script asserts the fingerprint on every later run.

## 4. Information allowed at time t

| Source | What | Point-in-time rule |
| --- | --- | --- |
| Open-Meteo Previous Runs API | GFS, ECMWF IFS 0.25°, ICON hourly 2 m temperature, `previous_day1` | Use only valid hours 00:00-20:00 local on D. A previous_day1 value is from a run initialized about 24 h before its valid time, so the latest one used was initialized by about 20:00 local on D-1 and was published well before 06:00 on D (see the leak check in 8). |
| IEM ASOS archive | Hourly station observations | Observations with timestamp <= t (gives the overnight temperatures, since the climate day starts at midnight). |
| RCC ACIS | Daily highs and normals | Daily highs up to D-1 only; normals any date. |
| IEM AFOS archive | NWS Area Forecast Discussions for the station's office | Products whose issuance time is <= t. |

No other data (news, web search, other forecast products) is available to any system.

## 5. Systems compared

| Id | System | Notes |
| --- | --- | --- |
| B0 | Climatology | Share of dev days at or above normal for that station and calendar month. |
| B1 | Model vote | Share of available models whose forecast high (max of hours 00-20 local) is >= threshold, shrunk as (k + 0.5) / (n + 1). |
| **B2** | **Statistical correction (main baseline)** | Logistic regression on: 3-model mean forecast high minus threshold, model spread (max - min), overnight observed max minus threshold, yesterday's forecast error of the model mean, station one-hot, sin/cos of day of year. Fit on dev only; evaluated on dev by forward chaining over 6 time blocks. A small and transparent version of the corrections meteorologists call MOS. |
| A1 | Agent | LangGraph agent on the Claude API with point-in-time tools: model forecasts, observations, normals, recent model errors, B2's probability, forecast discussions. Returns a probability as structured output. |
| A2 | Agent without discussions | A1 with the discussion tool removed (does the text add anything?). |
| A3 | Agent without B2 | A1 with the B2 tool removed (does it reason from the raw data?). |

The tool server, not the agent, enforces the time cutoff on every call and logs the newest timestamp it
returned; any timestamp after t counts as a leak and invalidates the run.

## 6. Metrics and decision rules

- Primary metric: Brier score. Secondary: log loss, reliability plot.
- Comparisons are paired on the same questions, with a 95% paired bootstrap CI (10,000 resamples) of the Brier
  difference. A system "beats" another only if the CI excludes 0.
- **Primary hypothesis H1:** A1 has a lower holdout Brier score than B2.
- **Secondary H2:** A1 beats A2 (the discussions help). **H3:** B2 beats B1 (correction helps).
- Agent samples (cost-limited): dev, 200 questions (20 per station, seed 7) for development; holdout,
  500 questions (50 per station, seed 11) for the one scored run. Baselines are also scored on all holdout
  questions. A secondary, pre-declared subset: questions where B2 is between 0.1 and 0.9 (the uncertain ones).
- Whatever the result, it is reported, including a null or negative one.

## 7. Data rules

Audited 2026-11-19 (data-rights review of each source's published terms; overall risk low; not legal advice).

| Source | Terms | Rules this project follows |
| --- | --- | --- |
| RCC ACIS (NOAA/NCEI observations, normals) | US-government work, public domain; ACIS disclaimer | One request at a time, one full-period request per station; cached. Recent observations are provisional, so outcomes are frozen with the fingerprint. |
| Open-Meteo Previous Runs API | Free API for non-commercial use only; data CC BY 4.0 (upstream ECMWF CC BY 4.0, DWD ICON CC BY 4.0, NOAA GFS public domain) | Under 600 calls/min, 5,000/hour, 10,000/day; monthly chunks, about 1 s between requests, exponential backoff on 429. Pin model names. No ads, sponsorships or paid tier while on the free API. |
| Iowa Environmental Mesonet (AFD text, ASOS) | Public domain; attribution appreciated; services heavily used | At most 1 request per second; date-range requests (one per office per month), never per product. |
| NWS forecast discussions | US-government work, public domain; no implied endorsement, not presented as official | Quoted verbatim, trims marked "[...]", office, product id and UTC issue time given; no NWS logos. |
| Anthropic API | Usage Policy | Public data only; outputs labeled as AI-generated with model name; outputs not used to train other models; key never committed. |

- Raw data and API caches are never committed (`data/` and the cache folder are gitignored); scripts download
  them on the user's machine and re-runs read from cache.
- The frozen question list (station, date, threshold, outcome, and the derived forecast high) may be committed.
  It carries a header with the CC BY 4.0 attribution and the changes made; the MIT license covers code only,
  and a NOTICE file says the data stays CC BY 4.0 or public domain.
- Forecast values come only from the hourly `temperature_2m_previous_dayN` variables. Plain `temperature_2m` and
  daily aggregates in that API come from the latest run, which would leak.
- README carries the full attribution block (Open-Meteo with link, ECMWF's required sentence, DWD, NOAA/NCEP,
  NCEI via ACIS, IEM) and a disclaimer: not affiliated with or endorsed by any provider or Anthropic; not an
  official forecast; not for safety, operational or betting use.
- All downloaders send a generic User-Agent (`forecast-agent-bench/0.1`), with no personal email.
- No personal data is collected.

## 8. Checks before any agent run

1. **Leak check on forecasts:** for Oct-Nov 2026, where Open-Meteo's Single Runs API has exact model runs,
   confirm that each previous_day1 value used equals a run initialized at or before 23:00 local on D-1. If not,
   switch to previous_day2 and record the change here.
2. Spot-check 20 questions by hand against the NWS climate reports.
3. Confirm the overnight-observation window matches the climate day (local standard time).
4. Baseline sanity: B2 must beat B0 on dev; if not, the feature pipeline is suspected broken.

## Change log

- 2026-11-19: frozen. (Before the freeze, section 8 was renamed from "before freezing" to "before any agent run",
  because its checks need downloaded data.)
