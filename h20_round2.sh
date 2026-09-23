#!/bin/bash
# Round 2 on H20: zero-overhead damped / frequency-split extrapolation vs zero-order reuse.
set -u
cd /data/viewdit
PY=/data/viewdit/env/py/bin/python
S=/data/viewdit/experiments/h20_cache_corrector.py
LOG=/data/viewdit/logs
until grep -q PIPELINE_RESUME_DONE $LOG/pipeline.log; do sleep 60; done
echo "round2 start $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S eval --part freq --shard 0 --nshards 2 > $LOG/eval_freq0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S eval --part freq --shard 1 --nshards 2 > $LOG/eval_freq1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "round2 eval done $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S score --shard 0 --nshards 2 > $LOG/score_r2_0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S score --shard 1 --nshards 2 > $LOG/score_r2_1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "ROUND2_DONE $(date '+%F %T')" >> $LOG/pipeline.log
