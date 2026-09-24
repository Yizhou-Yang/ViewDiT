#!/bin/zsh
D=/Users/yizhouyang/With/20260921/m6sk/v100_runs
while true; do
  ssh -o ConnectTimeout=20 viewdit-v100 'cd /root/viewdit && tar czf - --exclude="*.pt" results/video/seven_* logs experiments/video/seven_*.py 2>/dev/null' > $D/pull.tgz 2>>$D/sync.err && tar xzf $D/pull.tgz -C $D 2>>$D/sync.err
  rm -f $D/pull.tgz
  rsync -az $D/ any4:/root/viewdit_v100_backup_20260924/ 2>>$D/sync.err
  echo "$(date) synced" >> $D/sync.log
  sleep 600
done
