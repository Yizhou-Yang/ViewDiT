#!/bin/zsh
# Every 30 min: commit small H20 artifacts (jsonl, summaries, logs) to GitHub. Large tensors are gitignored.
cd /Users/yizhouyang/With/20260921/m6sk || exit 1
while true; do
  ssh -o BatchMode=yes -o ConnectTimeout=20 -p 36000 root@28.39.86.81 \
    '/data/viewdit/env/py/bin/python /data/viewdit/experiments/h20_analyze.py /data/viewdit/results > /data/viewdit/results/SUMMARY.txt 2>&1' || true
  sleep 60
  git add h20_runs/sync.log h20_runs/pod_mirror/logs h20_runs/pod_mirror/results h20_runs/pod_mirror/results_f49 h20_runs/pod_mirror/experiments 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "H20 auto-archive $(date '+%F %T')" && git push -q origin main \
      && echo "$(date '+%F %T') push ok" >> h20_runs/git_push.log || echo "$(date '+%F %T') push failed" >> h20_runs/git_push.log
  fi
  sleep 1740
done
