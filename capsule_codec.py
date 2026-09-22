#!/usr/bin/env python3
"""Versioned exact float32 capsule codec; shared model/schema required."""
import hashlib
import json
import struct
import zlib
from pathlib import Path
import numpy as np
import torch
MAGIC=b'VCAP0001'


def save_capsule(path, metadata, moments):
    layout=[];arrays=[]
    for (name,call),values in sorted(moments.items()):
        for slot,value in enumerate(values):
            array=value.detach().cpu().float().contiguous().numpy().astype('<f4',copy=False)
            layout.append({'name':name,'call':call,'slot':slot,'shape':list(array.shape),'count':array.size})
            arrays.append(array.tobytes())
    raw=b''.join(arrays)
    header=json.dumps({'metadata':metadata,'layout':layout,'dtype':'little-endian-float32','sha256':hashlib.sha256(raw).hexdigest()},separators=(',',':')).encode()
    payload=zlib.compress(struct.pack('<I',len(header))+header+raw,9)
    Path(path).write_bytes(MAGIC+payload)


def load_capsule(path):
    data=Path(path).read_bytes();assert data[:8]==MAGIC
    unpacked=zlib.decompress(data[8:]);n=struct.unpack('<I',unpacked[:4])[0]
    header=json.loads(unpacked[4:4+n]);raw=unpacked[4+n:]
    assert hashlib.sha256(raw).hexdigest()==header['sha256']
    offset=0;out={}
    for item in header['layout']:
        length=item['count']*4
        array=np.frombuffer(raw[offset:offset+length],dtype='<f4').copy().reshape(item['shape'])
        out.setdefault((item['name'],item['call']),[None,None])[item['slot']]=torch.from_numpy(array)
        offset+=length
    assert offset==len(raw)
    return header['metadata'],{k:tuple(v) for k,v in out.items()}


def main():
    root=Path('/root/viewdit/results/video/cog_semantic_capsule_restore')
    rows=[]
    for rnd in [1,2]:
        source=root/f'round{rnd}_capsule.pt'
        d=torch.load(source,map_location='cpu',weights_only=True)
        target=root/f'round{rnd}.vcap'
        save_capsule(target,d['schema'],d['moments'])
        meta,decoded=load_capsule(target)
        assert meta==d['schema']
        assert decoded.keys()==d['moments'].keys()
        assert all(torch.equal(a,b) for k,v in decoded.items() for a,b in zip(v,d['moments'][k]))
        rows.append({'round':rnd,'original_torch_file_bytes':source.stat().st_size,'vcap_file_bytes':target.stat().st_size,'tensor_payload_bytes':sum(t.numel()*4 for v in decoded.values() for t in v),'roundtrip_exact':True})
    (root/'codec_stats.json').write_text(json.dumps({'complete':True,'rows':rows},indent=2));print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
