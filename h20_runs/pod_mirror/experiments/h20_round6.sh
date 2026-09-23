#!/bin/bash
# Round 6: per-step sensitivity profile on CAL (seed A), then sensitivity-allocated reuse schedules on TEST,
# each matched in reuse count / max consecutive reuse to a uniform-late schedule from rounds 3 and 5.
set -u
cd /data/viewdit
PY=/data/viewdit/env/py/bin/python
S=/data/viewdit/experiments/h20_cache_corrector.py
LOG=/data/viewdit/logs
until grep -q ROUND5_DONE $LOG/pipeline.log; do sleep 60; done
echo "round6 start $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S profile --shard 0 --nshards 2 > $LOG/profile0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S profile --shard 1 --nshards 2 > $LOG/profile1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "round6 profile done $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S eval --part alloc --shard 0 --nshards 2 > $LOG/eval_alloc0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S eval --part alloc --shard 1 --nshards 2 > $LOG/eval_alloc1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "round6 eval done $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S score --shard 0 --nshards 2 > $LOG/score_r6_0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S score --shard 1 --nshards 2 > $LOG/score_r6_1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "ROUND6_DONE $(date '+%F %T')" >> $LOG/pipeline.log
