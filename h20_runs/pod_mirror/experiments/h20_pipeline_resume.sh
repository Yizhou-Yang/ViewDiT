#!/bin/bash
# Resume after env incident (2026-09-23 22:40-22:55: pyiqa install broke diffusers import; env restored to frozen list).
cd /data/viewdit/experiments
PY=/data/viewdit/env/py/bin/python; S=h20_cache_corrector.py; L=/data/viewdit/logs
echo "resume $(date '+%F %T')" >> $L/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S eval --part ridge --shard 0 --nshards 2 > $L/eval_ridge0_r.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $S eval --part ridge --shard 1 --nshards 2 > $L/eval_ridge1_r.log 2>&1 &
wait; echo "eval ridge resumed done $(date '+%F %T')" >> $L/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S score --shard 0 --nshards 2 > $L/score0_r.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $S score --shard 1 --nshards 2 > $L/score1_r.log 2>&1 &
wait; echo "PIPELINE_RESUME_DONE $(date '+%F %T')" >> $L/pipeline.log
