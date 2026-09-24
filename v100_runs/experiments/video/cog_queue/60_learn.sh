set -e
cd /root/viewdit/experiments/video
cp cog_combo_next3.py cog_combo.py
PY=/root/miniconda3/bin/python
$PY cog_combo.py calib --phase learn --p0 0 --nprompt 4 --seeds 5000 --only S4_OSx0_d0.5,S4_OSx0_d0.5+L4,S4_OSv1_d0.5+L4,S4L_OS3x0
$PY cog_combo.py gen --phase learn --p0 8 --nprompt 8 --seeds 3000
$PY cog_combo.py score --phase learn
$PY cog_combo.py summary --phase learn
