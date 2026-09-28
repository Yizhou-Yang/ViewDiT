set -e
cd /root/viewdit/experiments/video
cp cog_combo_next3.py cog_combo.py
PY=/root/miniconda3/bin/python
$PY cog_combo.py gen --phase confirm2 --p0 8 --nprompt 8 --seeds 3000 4000
$PY cog_combo.py score --phase confirm2
$PY cog_combo.py summary --phase confirm2
