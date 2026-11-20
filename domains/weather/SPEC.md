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

## 9. Precipitation track (added 2026-11-19, before any outcome was analyzed)

A second, separately scored question set through the same pipeline. Sections 1-8 still govern the
temperature track unchanged; where this section is silent, the same rules apply (stations, dates, split,
decision time t, point-in-time enforcement, leak audit, bootstrap, one-shot holdout, data rules).

**Question:** at 06:00 local time on day D, will measurable precipitation (at least 0.01 in) be recorded at
station S for climate day D?

- **Outcome:** ACIS daily precipitation `pcpn` for the climate day. `1` if >= 0.01 in; trace (`T`) and 0.00
  are `0`. Missing values and values flagged as multi-day accumulations are excluded and counted.
- **Already decided:** questions where the station's hourly ASOS reports show >= 0.01 in accumulated between
  the start of the climate day and t are excluded and counted, because the answer is already known at t.

**Information allowed at t (in addition to section 4):**

| Source | What | Point-in-time rule |
| --- | --- | --- |
| Open-Meteo Previous Runs API | Hourly precipitation for GFS, ECMWF IFS 0.25°, ICON | `previous_day1` for valid hours 00:00-20:59 local standard time on D, `previous_day2` for 21:00-23:59. Summed by the build script. |
| IEM MOS archive | NWS GFS MOS guidance (6-hour precipitation probabilities `p06`, 12-hour `p12`) | Only the 00Z run of day D (published by about 04Z, before t at every station). |
| IEM ASOS archive | Hourly precipitation `p01i` | Reports with timestamp <= t. |

**Systems:**

| Id | System | Notes |
| --- | --- | --- |
| P0 | Climatology | Share of dev days with measurable precipitation for that station and calendar month. |
| P1 | Model vote | Share of available models whose day total is >= 0.01 in, shrunk as (k + 0.5) / (n + 1). |
| **P2** | **Statistical correction (main baseline)** | Logistic regression on: log(1 + day total) for each model, number of models >= 0.01 in and >= 0.10 in, station one-hot, sin/cos of day of year. Fit on dev only; forward chaining over 6 time blocks on dev. |
| P3 | MOS-based | Logistic regression on the 00Z GFS MOS `p06` values whose periods overlap the climate day. Fit on dev only. Used only if the MOS archive covers at least 95% of the questions; otherwise dropped, and the change log says so. |
| A1p | Agent | As A1, with precipitation tools (model precipitation, MOS probabilities, overnight precipitation, P2's probability) and the forecast discussions. |
| A2p | Agent without discussions | A1p without the discussion tool. |
| A3p | Agent without P2 | A1p without the P2 tool. |

**Hypotheses (scored the same way as section 6):**
- **H4 (primary for this track):** A1p has a lower holdout Brier score than P2.
- **H5:** A1p beats A2p (the discussions help). **H6:** P2 beats P1. **H7:** A1p vs P3 (reported either way).
- The temperature and precipitation tracks are reported separately and together; a win on one track is
  never presented as a win on the other.
- Agent samples: dev 200 (seed 8), holdout 500 (50 per station, seed 12). Pre-declared subsets: P2 between
  0.1 and 0.9, and by station (arid stations have few wet days).

**Pre-run checks (added to section 8):** spot-check 20 precipitation outcomes against NWS climate reports;
confirm the multi-day-accumulation flag handling; confirm the MOS 00Z publication time and coverage.

**Data rules:** the IEM MOS archive is NWS guidance, US-government public domain, under the same IEM throttle
(at most 1 request per second per public IP, which all of the author's machines share).

## 10. Amendment A: contamination, stronger baselines, statistics, budget (2026-11-19)

Adopted after an independent design review, before any agent code existed and before any forecast had been
compared with an outcome. Where this section conflicts with sections 1-9, this section governs.

**10.1 Model and budget.**
- Every LLM system (A1-A3, A1p-A3p, the extraction baselines and the recall probe) uses the pinned model
  `claude-haiku-5-5`, at its default effort, one run per question, at most 8 tool calls per question.
- Exploratory cost-versus-quality comparison (H8): `claude-sonnet-5-5` and `claude-opus-5-5` each run A1/A1p on
  the same 100 post-cutoff holdout questions per track (stratified by station, seed 13) that Haiku also answers.
  Reported as Brier score against cost per 1,000 forecasts for the three models, with the baselines as
  reference lines.
- Hard budget: $30 of API spend for all runs combined. The runner computes spend from the token counts the API
  returns and stops at the cap. Spending priority if the cap binds: (1) A1/A1p on the primary holdout,
  (2) A2/A3/A2p/A3p on the primary holdout, (3) extraction baselines, (4) dev runs, (5) the Opus comparison,
  (6) the secondary holdout. Anything not run is reported as not run.

**10.2 Training-data contamination.**
- The **primary holdout** runs from the first day of the month after the pinned model's published training-data
  cutoff to 2026-11-15, and starts no earlier than 2026-07-01. The cutoff and its source are recorded in the
  change log before any agent run.
- The rest of the 2026 holdout is the **secondary holdout**, reported separately.
- **Recall probe (R):** the model with no tools, given only station, date and threshold, on 200 dev and 200
  primary-holdout questions per track. Its Brier score on dev vs. post-cutoff shows how much it remembers.

**10.3 Baselines.**
- B2 and P2 are renamed **SSP** (simple statistical post-processing). They are not MOS, and the write-up does not
  call them MOS.
- **T3 (MOS-based, temperature):** logistic regression on the 00Z GFS MOS daytime maximum (`n_x`) valid for day D
  minus the threshold. Fit on dev. Used only if the MOS archive covers at least 95% of questions.
- **E and Ep (extraction baselines):** one model call reads the latest discussion issued at or before t and
  returns only the high temperature it states for the station's area and the precipitation chance it states
  (either may be null). A logistic calibration fitted on a dev sample of 1,500 questions maps these to
  probabilities; when the value is null, SSP's probability is used. One call serves both tracks.
- The agents' tools add the MOS values (temperature and precipitation).

**10.4 Temperature questions already decided.** Temperature questions whose hourly ASOS temperatures between
the start of the climate day and t already reach the threshold are excluded and counted, as section 9 does for
precipitation.

**10.5 Statistics.**
- **Co-primary hypotheses:** H1 (A1 vs SSP, temperature) and H4 (A1p vs SSP, precipitation), each on all
  questions in the primary holdout. The family-wise error rate is 0.05 with a Holm correction, using two-sided
  bootstrap p-values.
- All other comparisons, subsets and the secondary holdout are **exploratory**, reported with 95% intervals and
  labeled as exploratory.
- **Block bootstrap:** resample whole dates (all stations on a date together), 10,000 resamples. This replaces
  the question-level bootstrap of section 6.
- **Sample sizes:** A1-A3 and A1p-A3p run on every primary-holdout question (no subsample). The dev runs
  (200 per track) are used to estimate the standard deviation of the paired difference; the minimum detectable
  effect for the primary tests is computed from it and recorded before the holdout run.
- Reported alongside Brier: the Brier skill score against climatology and the reliability / resolution
  decomposition, overall and by station.

**10.6 Run protocol.**
- A failed question (API error, timeout, invalid output) gets SSP's probability, is still scored, and every
  failure is counted and reported.
- Each question logs input, output and cache tokens, cost and latency.

**10.7 Leak-check rule (as implemented in `checks.py`).** For 5 stations (NYC, ORD, DEN, PHX, LAX), weekly days
from 2026-08-03 to 2026-11-15, every archived value used is matched to the exact model runs initialized from
12Z two days before D through 06Z on D. The check passes only if at least 90% of values match a run and every
matched run was initialized at least 6 hours before t (taking the latest run when several match). Section 8,
item 1 is replaced by this rule.

## Change log

- 2026-11-19: frozen. (Before the freeze, section 8 was renamed from "before freezing" to "before any agent run",
  because its checks need downloaded data.)
- 2026-11-19 (later the same day): added section 9, a precipitation question set, at the author's decision.
  At the time of this change, temperature forecasts and part of the forecast discussions had been downloaded,
  but no forecast had been compared with any outcome and no baseline had been fitted. Outcome data seen so
  far, all from availability checks: completeness counts per station, Seattle's share of wet days (35%), and
  a few individual days printed while testing the APIs. The temperature track (sections 1-8) is unchanged.
- 2026-11-19 (later): added section 10 (Amendment A) after an independent design review, approved by the
  author. Model pinned to claude-haiku-5-5 with a $30 hard cap; post-cutoff primary holdout; SSP naming;
  MOS and extraction baselines; temperature "already decided" exclusion; co-primary H1/H4 with Holm;
  date-block bootstrap; failure fallback; leak-check rule aligned with the code. Data state unchanged since the
  previous entry: still no forecast compared with any outcome and no model fitted.
- 2026-11-19: pinned model cutoff recorded (10.2). Anthropic's models overview
  (https://platform.claude.com/docs/en/about-claude/models/overview, read 2026-11-19) lists claude-haiku-5-5
  and claude-opus-5-5 with "Training data cutoff: Jun 2026" and "Reliable knowledge cutoff: Jun 2026".
  Primary holdout = 2026-07-01 to 2026-11-15; secondary holdout = 2026-01-01 to 2026-06-30.
- 2026-11-19: H8 extended at the author's decision to include claude-sonnet-5-5 (same page lists its cutoff as
  Jun 2026), giving a three-model cost-versus-quality comparison on the same questions. Budget cap unchanged
  ($30); the comparison stays at spending priority 5. No agent had run yet.
