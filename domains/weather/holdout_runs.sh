#!/bin/sh
# One-time holdout runs, in the SPEC 10.1 spending order. Every run shares the $55 ledger (data/runs/spend.json);
# a run that would pass the cap stops, and later runs stop at their first question. Resumable: re-running skips
# questions already answered. Scoring happens separately, once: python -m domains.weather.score --holdout-once
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
RUN="$PY -u -m domains.weather.run_agent"
L=logs/holdout
mkdir -p $L
echo "start $(date -Is)" > $L/chain.log

# 1. main agent, every primary-holdout question
$RUN --track temp   --system A1  --period holdout_primary --all --label holdout > $L/A1_temp.log 2>&1
$RUN --track precip --system A1p --period holdout_primary --all --label holdout > $L/A1p_precip.log 2>&1
echo "1 done $(date -Is)" >> $L/chain.log

# 2. ablations: no forecast discussion (A2), no statistical baseline (A3)
$RUN --track temp   --system A2  --period holdout_primary --all --label holdout > $L/A2_temp.log 2>&1
$RUN --track precip --system A2p --period holdout_primary --all --label holdout > $L/A2p_precip.log 2>&1
$RUN --track temp   --system A3  --period holdout_primary --all --label holdout > $L/A3_temp.log 2>&1
$RUN --track precip --system A3p --period holdout_primary --all --label holdout > $L/A3p_precip.log 2>&1
echo "2 done $(date -Is)" >> $L/chain.log

# 3. extraction baseline (+ calibration on dev) and the recall probe
$PY -u -m domains.weather.llm_baselines extract > $L/extract.log 2>&1
$PY -u -m domains.weather.llm_baselines fit >> $L/extract.log 2>&1
$PY -u -m domains.weather.llm_baselines probe > $L/probe.log 2>&1
echo "3 done $(date -Is)" >> $L/chain.log

# 5. H8 cost-versus-quality: Sonnet and Opus on the same 100 primary questions per track (seed 13)
for M in claude-sonnet-5-5 claude-opus-5-5; do
  EST=0.06; [ "$M" = claude-opus-5-5 ] && EST=0.12
  $RUN --track temp   --system A1  --period holdout_primary --n 100 --seed 13 --model $M --est-cost $EST --label h8 > $L/h8_temp_$M.log 2>&1
  $RUN --track precip --system A1p --period holdout_primary --n 100 --seed 13 --model $M --est-cost $EST --label h8 > $L/h8_precip_$M.log 2>&1
done
echo "5 done $(date -Is)" >> $L/chain.log

# 6. secondary holdout (Jan-Jun 2026, inside the model's training period), main agent only
$RUN --track temp   --system A1  --period holdout_secondary --all --label holdout > $L/A1_temp_secondary.log 2>&1
$RUN --track precip --system A1p --period holdout_secondary --all --label holdout > $L/A1p_precip_secondary.log 2>&1
echo "6 done $(date -Is)" >> $L/chain.log
echo "CHAIN DONE $(date -Is)" >> $L/chain.log
