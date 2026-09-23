#!/bin/zsh
set -u
LOCAL=/Users/yizhouyang/With/20260921/m6sk/h20_runs
POD="root@28.39.86.81"
mkdir -p "$LOCAL"
while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  rsync -az --timeout=120 -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -p 36000" \
    --exclude 'weights/' --exclude 'env/' --exclude '*.safetensors' --exclude '*.pth' \
    --max-size=300m "$POD:/data/viewdit/" "$LOCAL/pod_mirror/" \
    && echo "$ts pull ok" >> "$LOCAL/sync.log" || echo "$ts pull failed" >> "$LOCAL/sync.log"
  rsync -az --timeout=120 -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
    "$LOCAL/" any4:/root/viewdit_h20_backup/ \
    && echo "$ts dev4 ok" >> "$LOCAL/sync.log" || echo "$ts dev4 failed" >> "$LOCAL/sync.log"
  sleep 600
done
