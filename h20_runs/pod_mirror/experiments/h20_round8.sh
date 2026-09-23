#!/bin/bash
# Round 8: per-block sensitivity profile on CAL, then block-allocated reuse at matched call budgets
# (BA_sens vs reversed-allocation control BA_rev vs uniform U / L4).
set -u
cd /data/viewdit
PY=/data/viewdit/env/py/bin/python
S=/data/viewdit/experiments/h20_cache_corrector.py
LOG=/data/viewdit/logs
until grep -q ROUND7_DONE $LOG/pipeline.log; do sleep 60; done
echo "round8 start $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S bprofile --shard 0 --nshards 2 > $LOG/bprofile0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S bprofile --shard 1 --nshards 2 > $LOG/bprofile1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "round8 bprofile done $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S eval --part blockalloc --shard 0 --nshards 2 > $LOG/eval_balloc0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S eval --part blockalloc --shard 1 --nshards 2 > $LOG/eval_balloc1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "round8 eval done $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S score --shard 0 --nshards 2 > $LOG/score_r8_0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S score --shard 1 --nshards 2 > $LOG/score_r8_1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "ROUND8_DONE $(date '+%F %T')" >> $LOG/pipeline.log
