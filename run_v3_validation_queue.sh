#!/bin/bash
set -eu
for i in $(seq 1 180); do
    if /root/miniconda3/bin/python -c 'import json,pathlib; p=pathlib.Path("/root/viewdit/data/v3_validation_data"); d=json.loads((p/"manifest.json").read_text()); assert all((p/r["path"]).exists() and (p/r["path"]).stat().st_size==r["bytes"] for r in d["files"])' 2>/dev/null; then
        if pgrep -f '^/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_transport_validation.py' >/dev/null; then
            echo validation_already_running
            exit 0
        fi
        exec /root/miniconda3/bin/python /root/viewdit/experiments/video/cog_transport_validation.py
    fi
    sleep 10
done
echo data_wait_timeout
exit 2
