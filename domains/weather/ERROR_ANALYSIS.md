# Error analysis: why the forecast discussions hurt the temperature agent

Exploratory and post-hoc: done after the one-time holdout scoring (2026-11-20); nothing here changes a scored result.
Code: `python -m domains.weather.error_analysis quant` and `... traces 15`. "Harm" = Brier(A1, with discussions)
minus Brier(A2, without), per question; positive means reading the discussion made the answer worse.

## Summary

On temperature the agent with discussions (A1) scored 0.0796 against 0.0706 without them (A2): +0.0091 per question,
+12.0 Brier in total over 1,321 primary-holdout questions. Three patterns are associated with most of it (exploratory: keyword patterns plus reading the agent's reasoning).

1. **Area-wide statements applied to a single station.** NWS discussions describe a whole forecast area (for example, highs
   several degrees below normal, or highs in the 70s over most of the area). The agent mapped those onto one airport's
   normal. The 143 answers (11%) that cite such a statement carry **42% of the total harm** (+5.1).
2. **Routine features read as news.** At Los Angeles (LAX) a marine layer is the normal summer state and is already in
   the models and the statistical baseline. In the reviewed answers the agent read the discussion's marine-layer and coastal-eddy
   wording as a reason to cool the forecast. LAX alone is **53% of the total harm** (+6.4 from 129 questions), although the high reached normal on 84%
   of LAX days in Jul-Nov 2026.
3. **Double counting.** Forecasters write the discussion from much the same model guidance the baseline uses (a plausible mechanism, not tested here). In 45% of the
   most-harmed answers the agent says it trusts the text over the models or the baseline ("I weight the forecasters'
   written reasoning more heavily"; "the statistical baseline cannot read the discussion"), against 22% of the
   most-helped. The prompt's instruction to move away from the baseline "only for a concrete reason" did not stop
   this: nearly every discussion contains something that reads as concrete.

## How the discussion moved answers

| Measure | Value |
| --- | --- |
| Answers moved meaningfully (abs. harm > 0.01) | 451 (34%); 62% of them made worse |
| Moved toward 50% (hedged) | 41% of answers; +8.8 total harm |
| Moved away from 50% (bolder) | +3.1 total harm |
| Flipped to the other side of 50% | 4% of answers; +5.6 total harm |
| Discussion pushed the probability **down** (n = 241) | outcome was NO only 24% of the time; average 0.70 to 0.57 |
| Discussion pushed the probability **up** (n = 66) | outcome was YES 64% of the time; average 0.56 to 0.66 |

Downward pushes, mostly "cooler than normal" language, were systematically wrong; upward pushes were roughly right.

## Where

| Station | Total harm | Mean per question |
| --- | --- | --- |
| Los Angeles (LAX) | +6.39 | +0.050 |
| Atlanta | +1.64 | +0.012 |
| Phoenix | +0.92 | +0.007 |
| Boston | +0.88 | +0.007 |
| Miami, Seattle, Denver, New York, Chicago | +0.09 to +0.68 each | small |
| Dallas-Fort Worth | -0.19 | (discussions helped slightly) |

By month, harm peaks in September (+0.018 per question) and turns slightly helpful in November (-0.004). Without LAX the
harm halves (+0.0047 per question) but stays positive.

## Failure-mode language (keyword heuristic on the agent's stated factors and reasoning)

| Language | 100 most harmed | 100 most helped | All |
| --- | --- | --- | --- |
| Area-wide "below/above normal" statement | 22% | 11% | 11% |
| Marine layer / low clouds / fog | 38% | 22% | 16% |
| Trusts the text over models or baseline | 45% | 22% | 18% |
| Fronts / clouds / storms | 70% | 78% | 79% |

Fronts and storms are not a failure signal by themselves: the agent mentions them about as often when it is right.

## Examples (Claude Haiku 5.5 output, quoted from the agent's answers)

- **LAX, Aug 12, normal 76°F, reached.** SSP 0.75, without discussion 0.72, with discussion **0.12**: "The forecast
  office expects the high at LAX to fall well below the 76°F normal ... Reaching 76°F would need the marine layer to
  clear early ... which the discussion does not expect."
- **LAX, Aug 31, normal 77°F, reached.** SSP 0.90, without 0.80, with **0.30**: "I weight the forecasters' written
  reasoning more heavily because they have more information than the raw model output, and the statistical baseline
  (0.899) appears not to reflect the discussed cooling."
- **Chicago, Sep 9, normal 78°F, reached.** SSP 0.94, without 0.85, with **0.40**: the discussion's area-wide statement
  about highs in the 70s was applied to O'Hare.

## What this suggests (to test on new days, not on this spent holdout)

1. Tell the agent that discussions describe an area and mostly restate model guidance; use them only for information
   the models lack (timing changes, events since the model run, local effects named for that station).
2. Give the agent the station's own recent behaviour against its normal (e.g. how often LAX has reached normal this
   month), so routine regimes aren't read as news.
3. Make "move away from the baseline" asymmetric in evidence: require a station-specific number, not an area adjective.
4. Simplest of all: for temperature, use the agent without discussions (A2), which tied SSP.

Any of these must be evaluated as a new pre-registered forward test on days after 2026-11-15; the holdout is spent.
