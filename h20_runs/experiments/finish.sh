while pgrep -f "phase bench " >/dev/null; do sleep 60; done
COG_MODEL=2b python3 h20_cog_bench.py summary --phase bench > ../logs/bench/2b_bench_summary.log 2>&1
while pgrep -f "phase bench5b" >/dev/null; do sleep 60; done
DISABLE_ADDMM_CUDA_LT=1 COG_MODEL=5b COG_OUT=/data/workspace/viewdit/results/h20_bench/5b_f49_s50_nolt COG_LAT=/data/workspace/viewdit/latents/h20_bench/5b_nolt COG_SUF=_g3 CUDA_VISIBLE_DEVICES=3 python3 h20_cog_bench.py score --phase bench5b > ../logs/bench/5b_score.log 2>&1
DISABLE_ADDMM_CUDA_LT=1 COG_MODEL=5b COG_OUT=/data/workspace/viewdit/results/h20_bench/5b_f49_s50_nolt python3 h20_cog_bench.py summary --phase bench5b > ../logs/bench/5b_bench_summary.log 2>&1
echo ALLDONE > ../logs/bench/ALLDONE
