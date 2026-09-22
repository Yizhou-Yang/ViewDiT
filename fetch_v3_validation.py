#!/usr/bin/env python3
"""Fixed third-round validation data and longer semantic inputs."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
ROOT=Path(__file__).resolve().parent/'v3_validation_data'
NEW=['car-shadow','cows','elephant','goat','hike','horsejump-low','kite-surf','libby']
SEMANTIC=['bear','blackswan','bus','boat']
BASE='https://huggingface.co/datasets/AlonzoLeeeooo/DAVIS-Edit/resolve/main/'
API='https://huggingface.co/api/datasets/AlonzoLeeeooo/DAVIS-Edit/tree/main/'


def fetch(row):
    p=ROOT/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            data=urllib.request.urlopen(BASE+row['path'],timeout=60).read()
            digest=hashlib.sha256(data).hexdigest()
            if row.get('lfs',{}).get('oid'):assert digest==row['lfs']['oid']
            p.write_bytes(data)
            return {'path':row['path'],'bytes':len(data),'sha256':digest}
        except Exception:
            if attempt==2:raise
            time.sleep(2)


def main():
    rows=[];lengths={}
    for clip in NEW+SEMANTIC:
        listing=json.load(urllib.request.urlopen(API+'JPEGImages/'+clip,timeout=40))
        frames=sorted([r for r in listing if r['path'].endswith('.jpg')],key=lambda r:r['path'])[:33 if clip in NEW else 49]
        assert len(frames)==(33 if clip in NEW else 49),(clip,len(frames))
        rows+=frames;lengths[clip]=len(frames)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,rows))
    (ROOT/'manifest.json').write_text(json.dumps({'new_validation':NEW,'semantic_development':SEMANTIC,'selection':'Fixed before new method validation outcomes; first33 or49 frames, lexicographic filenames. No outcome screening.','frames':lengths,'files':results},indent=2))
    print('COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
