#!/bin/bash
# Round 7: TeaCache-style dynamic step skipping baseline (accumulated rel-L1 of block-0 modulated input), all 30 blocks.
set -u
cd /data/viewdit
PY=/data/viewdit/env/py/bin/python
S=/data/viewdit/experiments/h20_cache_corrector.py
LOG=/data/viewdit/logs
until grep -q ROUND6_DONE $LOG/pipeline.log; do sleep 60; done
echo "round7 start $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S eval --part tea --shard 0 --nshards 2 > $LOG/eval_tea0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S eval --part tea --shard 1 --nshards 2 > $LOG/eval_tea1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "round7 eval done $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S score --shard 0 --nshards 2 > $LOG/score_r7_0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S score --shard 1 --nshards 2 > $LOG/score_r7_1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "ROUND7_DONE $(date '+%F %T')" >> $LOG/pipeline.log
