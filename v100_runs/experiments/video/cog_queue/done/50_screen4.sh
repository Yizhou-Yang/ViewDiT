set -e
cd /root/viewdit/experiments/video
cp cog_combo_next3.py cog_combo.py
/root/miniconda3/bin/python cog_combo.py gen --phase screen4 --nprompt 8 --seeds 3000
/root/miniconda3/bin/python cog_combo.py score --phase screen4 --nprompt 8 --seeds 3000
/root/miniconda3/bin/python cog_combo.py summary --phase screen4
