#!/bin/bash
# Unattended finalizer: after round8 / round9 finish, write analysis text files into results dirs
# (picked up by the local rsync loop -> local mirror -> dev4, and by the git loop -> GitHub).
set -u
cd /data/viewdit
PY=/data/viewdit/env/py/bin/python
E=/data/viewdit/experiments
LOG=/data/viewdit/logs
until grep -q ROUND8_DONE $LOG/pipeline.log; do sleep 60; done
$PY $E/h20_analyze.py results > results/SUMMARY_r8.txt 2>&1
$PY $E/h20_quality_bias.py results > results/QUALITY_DELTAS_r8.txt 2>&1
$PY $E/h20_pairs.py results BA_sens_450:BA_rev_450 BA_sens_450:L4_all_w10_k4 BA_sens_c480:BA_rev_c480 BA_sens_c480:U_c480 U_c480:L4_all_w10_k4 > results/PAIRS_r8.txt 2>&1
cp results/eval/blockalloc_meta.json results/blockalloc_meta_copy.json 2>/dev/null
echo "finalize r8 $(date '+%F %T')" >> $LOG/pipeline.log
until grep -q ROUND9_DONE $LOG/pipeline.log; do sleep 60; done
$PY $E/h20_analyze.py results_f49 > results_f49/SUMMARY_f49.txt 2>&1
$PY $E/h20_quality_bias.py results_f49 > results_f49/QUALITY_DELTAS_f49.txt 2>&1
$PY $E/h20_pairs.py results_f49 B2_zero_late:B2_zero L4_all_w10_k4:steps15 L4_all_w10_k4:tea0.1 > results_f49/PAIRS_f49.txt 2>&1
echo "FINALIZE_DONE $(date '+%F %T')" >> $LOG/pipeline.log
