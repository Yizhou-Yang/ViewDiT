#!/usr/bin/env python3
"""Fetch fixed unseen content, with all source hashes recorded."""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent / 'boundary_holdout'
CLIPS = ['bmx-trees', 'boat', 'bus', 'dog']
API = 'https://huggingface.co/api/datasets/AlonzoLeeeooo/DAVIS-Edit/tree/main/'
BASE = 'https://huggingface.co/datasets/AlonzoLeeeooo/DAVIS-Edit/resolve/main/'


def fetch(item):
    data = urllib.request.urlopen(BASE + item['path'], timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if item.get('lfs', {}).get('oid'):
        assert digest == item['lfs']['oid']
    path = ROOT / item['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'path': item['path'], 'sha256': digest, 'bytes': len(data)}


def main():
    items = []
    for clip in CLIPS:
        listing = json.load(urllib.request.urlopen(API + 'JPEGImages/' + clip, timeout=45))
        frames = sorted([x for x in listing if x['path'].endswith('.jpg')], key=lambda x: x['path'])[:17]
        assert len(frames) == 17
        items += frames
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, items))
    (ROOT / 'manifest.json').write_text(json.dumps({'source': BASE, 'clips': CLIPS, 'selection': 'Fixed before outcomes: bmx-trees, boat, bus, dog; first17 JPEG frames. Never used in earlier four-clip pilot.', 'files': files}, indent=2))
    print('VERIFIED', len(files), flush=True)


if __name__ == '__main__':
    main()
