#!/usr/bin/env python3
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path
ROOT=Path('/root/viewdit/data/davis_locality_pilot')
API='https://hf-mirror.com/api/datasets/AlonzoLeeeooo/DAVIS-Edit/tree/main/'
BASE='https://hf-mirror.com/datasets/AlonzoLeeeooo/DAVIS-Edit/resolve/main/'
CLIPS=['bear','blackswan','car-roundabout','camel']

def fetch(item):
    path=ROOT/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            data=urllib.request.urlopen(BASE+item['path'],timeout=45).read()
            expected=item.get('lfs',{}).get('oid')
            if expected:assert hashlib.sha256(data).hexdigest()==expected
            path.write_bytes(data)
            return {'path':item['path'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        except Exception:
            if attempt==2:raise
            time.sleep(2)

def main():
    ROOT.mkdir(parents=True,exist_ok=True);items=[]
    for clip in CLIPS:
        listing=json.load(urllib.request.urlopen(API+'JPEGImages/'+clip,timeout=45))
        files=sorted([x for x in listing if x['path'].endswith('.jpg')],key=lambda x:x['path'])[:17]
        assert len(files)==17
        items+=files
    with concurrent.futures.ThreadPoolExecutor(4) as pool:results=list(pool.map(fetch,items))
    (ROOT/'manifest.json').write_text(json.dumps({'source':'https://huggingface.co/datasets/AlonzoLeeeooo/DAVIS-Edit','clips':CLIPS,'selection':'first 17 frames in filename order; no outcome selection','files':results},indent=2))
    print('Downloaded and verified',len(results),'frames',flush=True)

if __name__=='__main__':main()
