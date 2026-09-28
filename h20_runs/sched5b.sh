#!/bin/sh
# H20 scheduler: 5b bench spread over GPUs as 2b shards finish; then summaries. Resumable (gen skips done rows).
cd /data/workspace/viewdit/experiments
L5=$(cat ../results/h20_bench/bench5b_list.txt)
O5=/data/workspace/viewdit/results/h20_bench/5b_f49_s50_nolt
LA5=/data/workspace/viewdit/latents/h20_bench/5b_nolt
run5() {
  g=$1; shift
  DISABLE_ADDMM_CUDA_LT=1 COG_MODEL=5b COG_OUT=$O5 COG_LAT=$LA5 COG_SUF=_g$g CUDA_VISIBLE_DEVICES=$g python3 h20_cog_bench.py gen --phase bench5b --plist "$@" --only $L5 >> ../logs/bench/5b_bench_g$g.log 2>&1
  DISABLE_ADDMM_CUDA_LT=1 COG_MODEL=5b COG_OUT=$O5 COG_LAT=$LA5 COG_SUF=_g$g CUDA_VISIBLE_DEVICES=$g python3 h20_cog_bench.py score --phase bench5b >> ../logs/bench/5b_score_g$g.log 2>&1
}
waitpid() { while kill -0 $1 2>/dev/null; do sleep 60; done; }
run5 3 4 &
( waitpid 11115; run5 0 8 ) &
( waitpid 11120; run5 1 12 ) &
( waitpid 11125; run5 2 0 ) &
wait
COG_MODEL=2b python3 h20_cog_bench.py summary --phase bench > ../logs/bench/2b_bench_summary.log 2>&1
COG_MODEL=5b COG_OUT=$O5 python3 h20_cog_bench.py summary --phase bench5b > ../logs/bench/5b_bench_summary.log 2>&1
echo ALLDONE > ../logs/bench/ALLDONE
