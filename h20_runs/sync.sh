#!/bin/zsh
# local <- H20 (tar over ssh, no latents/weights), then -> dev4; every 3rd loop push to GitHub.
R=/Users/yizhouyang/With/20260921/m6sk
D=$R/h20_runs
mkdir -p $D
n=0
while true; do
  ssh -o ConnectTimeout=20 h20b 'cd /data/workspace/viewdit && tar czf - --exclude="*.pt" results logs experiments 2>/dev/null' > $D/pull.tgz 2>>$D/sync.err && tar xzf $D/pull.tgz -C $D 2>>$D/sync.err
  rm -f $D/pull.tgz
  mkdir -p $D/v100_refq
  ssh -o ConnectTimeout=20 viewdit-v100 'cd /root/viewdit/refq && tar czf - --exclude="*.pt" --exclude=latents --exclude="lat_*" out sc_2b_f17 code 2>/dev/null' > $D/v100.tgz 2>>$D/sync.err && tar xzf $D/v100.tgz -C $D/v100_refq 2>>$D/sync.err
  rm -f $D/v100.tgz
  rsync -az $D/ any4:/root/viewdit_h20_backup_20260928/ 2>>$D/sync.err
  rsync -az $R/*.md $R/*.py any4:/root/viewdit_h20_backup_20260928/local_docs/ 2>>$D/sync.err
  echo "$(date) synced" >> $D/sync.log
  n=$((n+1))
  if [ $((n % 3)) -eq 0 ]; then
    cd $R && git add h20_runs/results h20_runs/v100_refq h20_runs/experiments h20_runs/logs h20_runs/sync.sh h20_cog_bench.py h20_refq.py h20_pareto.py h20_paired.py h20_fair_analyze.py *.md 2>>$D/sync.err
    git commit -qm "auto: H20 bench sync $(date +%F_%H:%M)" 2>>$D/sync.err && git push -q origin main >>$D/git_push.log 2>&1
    echo "$(date) git rc=$?" >> $D/sync.log
  fi
  sleep 600
done
