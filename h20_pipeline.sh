#!/bin/bash
# Orchestrates the full H20 study; every stage logs to /data/viewdit/logs and is resumable.
cd /data/viewdit/experiments
PY=/data/viewdit/env/py/bin/python
S=h20_cache_corrector.py
L=/data/viewdit/logs
gpu0() { for b in B1 B2 B3; do CUDA_VISIBLE_DEVICES=0 $PY $S fit --budget $b > $L/fit_$b.log 2>&1; echo "fit $b rc=$?" >> $L/pipeline.log; done; }
gpu1() { CUDA_VISIBLE_DEVICES=1 $PY $S eval --part base --shard 0 --nshards 1 > $L/eval_base.log 2>&1; echo "eval base rc=$?" >> $L/pipeline.log; }
echo "start $(date '+%F %T')" >> $L/pipeline.log
gpu0 & P0=$!
gpu1 & P1=$!
wait $P0 $P1
CUDA_VISIBLE_DEVICES=0 $PY $S eval --part ridge --shard 0 --nshards 2 > $L/eval_ridge0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $S eval --part ridge --shard 1 --nshards 2 > $L/eval_ridge1.log 2>&1 &
wait; echo "eval ridge done $(date '+%F %T')" >> $L/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S score --shard 0 --nshards 2 > $L/score0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $S score --shard 1 --nshards 2 > $L/score1.log 2>&1 &
wait; echo "PIPELINE_DONE $(date '+%F %T')" >> $L/pipeline.log
