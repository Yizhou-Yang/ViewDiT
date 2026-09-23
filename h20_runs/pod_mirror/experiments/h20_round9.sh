#!/bin/bash
# Round 9: generalization of the late-window schedule to 49 frames (13 latent frames), 8 TEST prompts x 1 seed.
# Schedule hyperparameters are frozen from 17-frame rounds (no re-tuning). Separate dirs results_f49/ latents_f49/.
set -u
cd /data/viewdit
PY=/data/viewdit/env/py/bin/python
S=/data/viewdit/experiments/h20_cache_corrector.py
LOG=/data/viewdit/logs
until grep -q ROUND8_DONE $LOG/pipeline.log; do sleep 60; done
echo "round9 start $(date '+%F %T')" >> $LOG/pipeline.log
export VIEWDIT_F=49 VIEWDIT_TAG=_f49 VIEWDIT_NTEST=8 VIEWDIT_NSEED=1
mkdir -p results_f49
cp results/embeddings.pt results_f49/embeddings.pt 2>/dev/null
CUDA_VISIBLE_DEVICES=0 $PY $S eval --part f49 --shard 0 --nshards 2 > $LOG/eval_f49_0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S eval --part f49 --shard 1 --nshards 2 > $LOG/eval_f49_1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "round9 eval done $(date '+%F %T')" >> $LOG/pipeline.log
CUDA_VISIBLE_DEVICES=0 $PY $S score --shard 0 --nshards 2 > $LOG/score_f49_0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $PY $S score --shard 1 --nshards 2 > $LOG/score_f49_1.log 2>&1 &
P1=$!
wait $P0 $P1
echo "ROUND9_DONE $(date '+%F %T')" >> $LOG/pipeline.log
