set -e
cd /root/viewdit/experiments/video
cp cog_combo_next3.py cog_combo.py
PY=/root/miniconda3/bin/python
$PY cog_combo.py gen --phase screen5 --nprompt 8 --seeds 3000
$PY cog_combo.py score --phase screen5
$PY cog_combo.py summary --phase screen5
