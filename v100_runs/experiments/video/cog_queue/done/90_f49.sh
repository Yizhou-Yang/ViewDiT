set -e
cd /root/viewdit/experiments/video
cp cog_combo_next3.py cog_combo.py
PY=/root/miniconda3/bin/python
export COG_F=49 COG_OUT=/root/viewdit/results/video/cog_combo_f49_20260924 COG_LAT=/root/viewdit/latents/cog_combo_f49_20260924
mkdir -p $COG_OUT && cp -n /root/viewdit/results/video/cog_combo_20260924/embeddings.pt $COG_OUT/
$PY cog_combo.py gen --phase confirm2 --p0 8 --nprompt 4 --seeds 3000
$PY cog_combo.py score --phase confirm2
$PY cog_combo.py summary --phase confirm2
