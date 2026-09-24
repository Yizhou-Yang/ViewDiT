#!/bin/zsh
# local <- V100 (tar over ssh, no *.pt), then -> dev4; every 30 min push small artifacts to GitHub.
R=/Users/yizhouyang/With/20260921/m6sk
D=$R/v100_runs
n=0
while true; do
  ssh -o ConnectTimeout=20 viewdit-v100 'cd /root/viewdit && tar czf - --exclude="*.pt" results/video/seven_* results/video/cog_combo_* logs experiments/video/seven_*.py experiments/video/cog_combo*.py experiments/video/cog_queue experiments/video/cog_queue_runner.sh 2>/dev/null' > $D/pull.tgz 2>>$D/sync.err && tar xzf $D/pull.tgz -C $D 2>>$D/sync.err
  rm -f $D/pull.tgz
  rsync -az $D/ any4:/root/viewdit_v100_backup_20260924/ 2>>$D/sync.err
  rsync -az $R/*.md $R/cog_combo*.py any4:/root/viewdit_v100_backup_20260924/local_docs/ 2>>$D/sync.err
  echo "$(date) synced" >> $D/sync.log
  n=$((n+1))
  if [ $((n % 3)) -eq 0 ]; then
    cd $R && git add v100_runs/results v100_runs/experiments v100_runs/logs cog_combo*.py cog_queue_runner.sh *.md v100_runs/sync.sh 2>>$D/sync.err
    git commit -qm "auto: V100 cog combo sync $(date +%F_%H:%M)" 2>>$D/sync.err && git push -q origin main >>$D/git_push.log 2>&1
    echo "$(date) git rc=$?" >> $D/sync.log
  fi
  sleep 600
done
