#!/bin/bash
# Serial GPU queue: runs every job file dropped into queue/ in name order; never leaves the GPU idle while jobs exist.
Q=/root/viewdit/experiments/video/cog_queue
mkdir -p $Q/done $Q/logs
cd /root/viewdit/experiments/video
while true; do
  j=$(ls $Q/*.sh 2>/dev/null | head -1)
  if [ -z "$j" ]; then sleep 60; continue; fi
  n=$(basename $j .sh)
  echo "$(date +%F_%T) START $n" >> $Q/queue.log
  bash $j > $Q/logs/$n.log 2>&1
  echo "$(date +%F_%T) END $n rc=$?" >> $Q/queue.log
  mv $j $Q/done/
done
