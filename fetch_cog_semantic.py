#!/usr/bin/env python3
"""Download official CogVideoX-2b files with LFS SHA256 verification."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request
ROOT = Path('/root/viewdit/weights/CogVideoX-2b')
META = Path('/root/viewdit/weights/cog_semantic_index.json')
BASE = 'https://hf-mirror.com/THUDM/CogVideoX-2b/resolve/main/'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def fetch(row):
    path = ROOT / row['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = row.get('lfs', {}).get('oid')
    if row['path'].startswith('vae/'):
        source = Path('/root/viewdit/weights/CogVAE') / path.name
        if source.exists() and not path.exists():
            path.symlink_to(source)
    if path.exists() and path.stat().st_size == row['size']:
        if expected:
            assert sha(path) == expected
        print('REUSE', row['path'], flush=True)
        return
    temp = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(BASE + row['path'], timeout=120) as r, temp.open('wb') as f:
        while True:
            block = r.read(8*1024*1024)
            if not block:
                break
            f.write(block)
    assert temp.stat().st_size == row['size'], row['path']
    if expected:
        assert sha(temp) == expected, row['path']
    os.replace(temp, path)
    print('VERIFIED', row['path'], flush=True)


def main():
    rows = json.loads(META.read_text())
    selected = [r for r in rows if r['type'] == 'file' and (r['path'].startswith(('text_encoder/', 'tokenizer/', 'transformer/', 'scheduler/', 'vae/')) or r['path'] in ['model_index.json', 'LICENSE'])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, selected))
    (ROOT / 'verified_manifest.json').write_text(json.dumps(selected, indent=2))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
