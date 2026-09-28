#!/bin/sh
O=/data/workspace/viewdit/results/h20_bench/2b_f49_s50_fairbench
while true; do
  left=0
  for g in 0 1 2 3; do
    n=$(wc -l < $O/gen_fairbench_g$g.jsonl)
    pid=$(pgrep -f "^python3 h20_cog_bench.py gen --phase fairbench --plist $((4+g)) ")
    if [ -n "$pid" ]; then
      left=1
      if [ "$n" -ge 66 ]; then kill $pid; echo "$(date) cap g$g at $n" >> /data/workspace/viewdit/logs/fair/cap.log; fi
    fi
  done
  [ $left -eq 0 ] && break
  sleep 20
done
echo "$(date) gen phase over" >> /data/workspace/viewdit/logs/fair/cap.log
