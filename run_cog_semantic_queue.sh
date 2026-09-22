#!/bin/bash
set -eu
for i in $(seq 1 540); do
    if /root/miniconda3/bin/python -c 'import json,pathlib; p=pathlib.Path("/root/viewdit/weights/CogVideoX-2b"); rows=json.loads((p/"verified_manifest.json").read_text()); assert all((p/r["path"]).exists() and (p/r["path"]).stat().st_size==r["size"] for r in rows); assert json.load(open("/root/viewdit/results/video/cog_context_capsule/stats.json"))["complete"]' 2>/dev/null; then
        /root/miniconda3/bin/python -c 'import json,pathlib,hashlib; p=pathlib.Path("/root/viewdit/weights/CogVideoX-2b"); rows=json.loads((p/"verified_manifest.json").read_text()); exec("for r in rows:\n if not r.get(\"lfs\"): continue\n h=hashlib.sha256()\n with (p/r[\"path\"]).open(\"rb\") as f:\n  for b in iter(lambda:f.read(8388608),b\"\"): h.update(b)\n assert h.hexdigest()==r[\"lfs\"][\"oid\"],r[\"path\"]\n print(\"REMOTE_VERIFIED\",r[\"path\"],flush=True)")'
        /root/miniconda3/bin/python /root/viewdit/experiments/video/cog_semantic_noninterference.py --clips bear --frames 17 --steps 4 --strength 1 --rounds 1 --out /root/viewdit/results/video/cog_semantic_smoke
        exec /root/miniconda3/bin/python /root/viewdit/experiments/video/cog_semantic_noninterference.py --clips bear --frames 49 --steps 50 --strength .6 --rounds 2 --out /root/viewdit/results/video/cog_semantic_bear49
    fi
    sleep 10
done
echo model_or_gpu_wait_timeout
exit 2
