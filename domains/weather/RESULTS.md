# Weather benchmark: holdout results (scored once, 2026-11-20 23:18 UTC)

All numbers are Brier scores (lower is better) with 95% date-block bootstrap intervals (10,000 resamples). Scored
by `python -m domains.weather.score --holdout-once` at commit 2966e51; code, prompts and tools frozen at tag
`holdout-freeze` (f93ee5d) before any holdout question was run. The agent is `claude-haiku-5-5` unless stated.
Primary holdout: 2026-07-01 to 2026-11-15, after the model's June 2026 training cutoff.

## Co-primary tests (SPEC 10.5, Holm at 0.05)

| Test | Agent | SSP baseline | Agent - SSP (95% CI) | p | Verdict |
| --- | --- | --- | --- | --- | --- |
| H1 temperature (n = 1,321) | 0.0796 | 0.0731 | +0.0065 (+0.0014, +0.0114) | 0.011 | Agent significantly **worse** |
| H4 precipitation (n = 1,271) | 0.0748 | 0.0821 | -0.0072 (-0.0120, -0.0025) | 0.002 | Agent significantly **better** |

H1 predicted the agent would beat SSP; the difference is significant in the opposite direction, so H1 is not
supported. H4 is supported.

## Primary holdout, all systems

| System | Temp Brier | Temp skill | Precip Brier | Precip skill |
| --- | --- | --- | --- | --- |
| Climatology | 0.2221 | 0 | 0.1340 | 0 |
| Model vote | 0.1119 | 0.50 | 0.1343 | 0.00 |
| SSP (statistical post-processing) | 0.0731 | 0.67 | 0.0821 | 0.39 |
| NWS MOS-based | 0.0816 | 0.63 | 0.0883 | 0.34 |
| Extraction (LLM reads the discussion's numbers) | 0.0873 | 0.61 | 0.0797 | 0.41 |
| **A1: agent, all tools** | **0.0796** | 0.64 | **0.0748** | 0.44 |
| A2: agent without the forecast discussion | 0.0706 | 0.68 | 0.0775 | 0.42 |
| A3: agent without the SSP tool | 0.1000 | 0.55 | 0.0771 | 0.43 |

Skill = 1 - Brier / Brier(climatology).

## Secondary comparisons (exploratory)

| Comparison | Temperature | Precipitation |
| --- | --- | --- |
| H2/H5: do the discussions help? (A1 - A2) | +0.0091 (+0.0057, +0.0125): they **hurt** | -0.0027 (-0.0064, +0.0010): no detectable effect |
| Baseline tool (A1 - A3) | -0.0203 (-0.0241, -0.0166): needed | -0.0022 (-0.0049, +0.0004): not needed |
| Agent vs NWS MOS-based (H7 for precip) | -0.0020 (-0.0094, +0.0053) | -0.0134 (-0.0208, -0.0067): agent better |
| Agent vs extraction | -0.0076 (-0.0145, -0.0009) | -0.0049 (-0.0097, -0.0002) |
| SSP vs model vote (H3/H6) | -0.0387: correction helps | -0.0522: correction helps |
| Uncertain questions only (SSP 0.1-0.9): A1 - SSP | +0.0029 (n = 445, CI includes 0) | -0.0243 (-0.0378, -0.0113), n = 407 |
| Secondary holdout Jan-Jun 2026: A1 - SSP | +0.0028 (-0.0008, +0.0061), n = 1,644 | -0.0042 (-0.0082, -0.0003), n = 1,614 |

Per station (temperature, A1 - SSP): Los Angeles is the largest loss, +0.0489 (+0.0266, +0.0727); Boston +0.0105;
the other eight are not distinguishable from 0. Precipitation: Atlanta -0.0222 and Dallas-Fort Worth -0.0178
better; the other eight are not distinguishable from 0.

## H8: cost versus quality (same 100 primary questions per track, exploratory)

| Model | Temp Brier | Precip Brier | $ per 1,000 forecasts (temp / precip) |
| --- | --- | --- | --- |
| SSP (reference, same questions) | 0.0698 | 0.0938 | ~0 |
| Haiku 5.5 | 0.0914 | 0.0846 | 2.22 / 1.84 |
| Sonnet 5.5 | 0.0720 | 0.0894 | 21.43 / 21.24 |
| Opus 5.5 | 0.0721 | 0.0806 | 80.67 / 64.75 |

On temperature, Sonnet and Opus are both significantly better than Haiku (about -0.019) and land within 0.0023 of SSP on these questions (not tested against SSP); on
precipitation, no model is distinguishable from Haiku. One Opus question failed and was scored with SSP's probability.

## Recall probe (no tools, no data; 200 questions per track and period)

| Track | Dev (in training data): probe / climatology | Primary holdout: probe / climatology |
| --- | --- | --- |
| Temperature | 0.2309 / 0.2386 | 0.2321 / 0.2321 |
| Precipitation | 0.1628 / 0.1703 | 0.1507 / 0.1516 |

The model with no tools scores about as well as climatology on both periods: a small edge on dev (~0.008) that
disappears after the training cutoff. Memorized weather is not a meaningful factor.

## Calibration (pre-registered secondary metrics, computed 2026-11-22 after the one-time scoring)

SPEC 6 and 10.5 named log loss, a reliability plot and the reliability / resolution decomposition as secondary
metrics. They were not computed at scoring time; `calibration.py` computes them read-only from the same run files
(the primary results above are unchanged). Brier = reliability - resolution + uncertainty (10 bins; lower
reliability error and higher resolution are better).

**Temperature** (n = 1,321, base rate 0.69, uncertainty 0.2140)

| System | Brier | Log loss | Skill vs climatology | Reliability (error) | Resolution |
| --- | --- | --- | --- | --- | --- |
| Agent, all tools | 0.0796 | 0.2685 | 0.642 | 0.0077 | 0.1422 |
| Agent, no discussion | 0.0706 | 0.2443 | 0.682 | 0.0054 | 0.1488 |
| SSP baseline | 0.0731 | 0.2318 | 0.671 | 0.0017 | 0.1424 |
| MOS-based | 0.0816 | 0.2602 | 0.633 | 0.0004 | 0.1326 |
| Climatology | 0.2221 | 0.6360 | 0.000 | 0.0175 | 0.0077 |

**Precipitation** (n = 1,271, base rate 0.185, uncertainty 0.1507)

| System | Brier | Log loss | Skill vs climatology | Reliability (error) | Resolution |
| --- | --- | --- | --- | --- | --- |
| Agent, all tools | 0.0748 | 0.2452 | 0.442 | 0.0032 | 0.0785 |
| Agent, no discussion | 0.0775 | 0.2533 | 0.422 | 0.0013 | 0.0735 |
| SSP baseline | 0.0821 | 0.2663 | 0.388 | 0.0029 | 0.0707 |
| MOS-based | 0.0883 | 0.2856 | 0.341 | 0.0028 | 0.0651 |
| Climatology | 0.1340 | 0.4289 | 0.000 | 0.0015 | 0.0174 |

- **Temperature: the agent ranks days as well as SSP but is miscalibrated.** Resolution 0.1422 vs 0.1424; reliability error 0.0077 vs 0.0017. The reliability curve sits above the diagonal between about 0.4 and 0.9: when the agent said around 50%, the high reached normal about 77% of the time. It was systematically too pessimistic, which matches the downward pushes in ERROR_ANALYSIS.md. Without the discussions its reliability error falls to 0.0054.
- **Precipitation: the gain is resolution.** 0.0785 vs 0.0707 for SSP, with similar reliability error (0.0032 vs 0.0029): the agent separates wet from dry days better.
- By station (temperature), the agent's reliability error is largest at LAX (0.0583 vs SSP 0.0345) and MIA (0.0340 vs 0.0063). Per-bin tables and every station: `calibration_results.json`. Chart: `docs/img/reliability.png`.

## Pre-registered checks and deviations

Every check the SPEC promised, what happened, and the evidence. Two were missed before the agent runs and done
afterwards; they can only confirm or question the data, not change a result.

| SPEC | Check | Status | Evidence |
| --- | --- | --- | --- |
| 8.1, 10.7 | Forecast leak check against exact model runs | Done before any agent run. v1 failed; diagnosed; v2 passed 6/6 | SPEC change log 2026-11-19; `checks.py`, `leak_diag.py` |
| 8.2, 9 | Spot-check 20 temperature and 20 precipitation outcomes against NWS climate reports | **Missed before the runs; done 2026-11-22 after scoring.** Final climate report found 20/20 and 20/20; observed values match ACIS 20/20 and 20/20; outcomes match 20/20 and 20/20 | `spot_check.py`, `spot_check_results.json` |
| 8.3 | Overnight window matches the climate day (local standard time) | Done in code before the runs, covered by tests | `climate_day_start_utc`; `tests/test_build_benchmark.py` (standard time all year); `tests/test_sources.py` |
| 8.4 | Baseline sanity: SSP beats climatology on dev | Passed (dev blocks 2-6: temperature 0.0796 vs 0.2400; precipitation 0.0734 vs 0.1369) | `score.py` dev scoring |
| 9 | Multi-day accumulation flags excluded | Done in code; 5 precipitation outcomes excluded (with missing values) | `parse_pcpn`; `tests/test_build_benchmark.py::test_parsers`; manifest |
| 9 | MOS 00Z run published before t | Not measured. Rule: the 00Z run becomes visible at 04Z (`MOS_LATENCY`); the earliest decision time at any station is 10Z, a 6-hour margin | `sources.py`; `tests/test_sources.py` |
| 9, 10 | MOS coverage of at least 95% of questions | Passed: 99.7% dev, 99.2% primary holdout (temperature); 99.6% / 99.3% (precipitation) | baselines tables |
| 6, 10.5 | Log loss, reliability plot, reliability / resolution decomposition | **Not computed at scoring time; computed 2026-11-22** | Calibration section above |

Other deviations and clarifications:

- **The scorer was written after the freeze** (commit 2966e51, before the one scoring run). It implements the tests
  SPEC 6, 9 and 10.5 already named; a lock file blocks a second run.
- **Minimum detectable effect.** The recorded temperature MDE (0.0123) used the dev paired-difference SD (0.1297,
  design effect 1.26) from 200 dev questions. The holdout differences varied less: the realized 95% CI half-width
  was about 0.005, so the observed +0.0065 was significant although smaller than the planned MDE.
- **The design review behind Amendment A was done by an AI reviewer** (a Claude agent prompted as a skeptical
  senior ML hiring manager), not a person; the author decided which changes to adopt.
- **The agent budget cap was raised from $30 to $55** before any holdout run (SPEC change log, 2026-11-20).
- SPEC change-log entries are grouped by topic; the 2026-11-19 leak-check entry follows 2026-11-20 entries. Git
  history (commit f1917b1) dates it.

## Runs and cost

10,000+ agent answers; 1 failure (an Opus question). Total API spend for the project: $41.99 of the $55 cap.
Haiku: about $2 per 1,000 forecasts, ~7 s per question, 5-6 tool calls.

## Note on the development runs

On 200 dev questions per track the agent looked better on temperature (-0.0151) and slightly worse on
precipitation (+0.0046), both inside their intervals. The holdout reversed both. That is why the decision rule
was fixed in advance and sized with a minimum detectable effect instead of trusting a small sample.
