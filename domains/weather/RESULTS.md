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

## Runs and cost

10,000+ agent answers; 1 failure (an Opus question). Total API spend for the project: $41.99 of the $55 cap.
Haiku: about $2 per 1,000 forecasts, ~7 s per question, 5-6 tool calls.

## Note on the development runs

On 200 dev questions per track the agent looked better on temperature (-0.0151) and slightly worse on
precipitation (+0.0046), both inside their intervals. The holdout reversed both. That is why the decision rule
was fixed in advance and sized with a minimum detectable effect instead of trusting a small sample.
