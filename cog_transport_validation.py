#!/usr/bin/env python3
"""Independent fixed-method validation on eight new contents and33 frames."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLCogVideoX
from cog_boundary_moment_transport import capture, BoundaryMomentTransport
ROOT=Path('/root/viewdit/data/v3_validation_data')
OUT=Path('/root/viewdit/results/video/cog_transport_validation')


@torch.no_grad()
def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for r in manifest['files']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    vae=AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report={'complete':False,'protocol':{'clips':manifest['new_validation'],'frames':33,'size':[128,224],'methods':['reference','dual'],'edits':['geometric1','encoded_brightness'],'boundary':29,'criterion_prefix':1e-6,'warning':'Eight new contents, fixed dual variant after4 development videos. Encoded brightness is not text semantic editing. Pixel splice has exact endpoint error0 and is included as an oracle, not excluded.'},'rows':[]}
    def save(row=None):
        if row is not None:report['rows'].append(row);print('ROW',json.dumps(row),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for clip in manifest['new_validation']:
        files=sorted(p for p in (ROOT/'JPEGImages'/clip).glob('*.jpg') if not p.name.startswith('.'))[:33]
        x=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((224,128)),dtype=np.float32)/127.5-1 for p in files])).permute(3,0,1,2).unsqueeze(0).cuda()
        z=vae.encode(x).latent_dist.mode();base,rs=capture(vae,z)
        field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5;field=field/field.square().mean().sqrt()*z.square().mean().sqrt()
        brighter=x.clone();brighter[:,:,29:]=(brighter[:,:,29:]+.5).clamp(-1,1)
        zb=vae.encode(brighter).latent_dist.mode()
        for edit in report['protocol']['edits']:
            ze=z.clone();ze[:,:,-1]=z[:,:,-1]+field[:,:,-1] if edit=='geometric1' else zb[:,:,-1]
            raw,es=capture(vae,ze);denom=(raw[:,:,29:]-base[:,:,29:]).square().mean().sqrt().clamp(min=1e-12)
            videos={'base':base.cpu(),'raw':raw.cpu()}
            pixel=raw.clone();pixel[:,:,:29]=base[:,:,:29];videos['pixel_splice']=pixel.cpu()
            for method in ['reference','dual']:
                control=BoundaryMomentTransport(vae.decoder,rs,es,target_call=3,mode=method)
                try:control.reset();y=vae.decode(ze).sample
                finally:control.close()
                noop=BoundaryMomentTransport(vae.decoder,rs,rs,target_call=3,mode=method)
                try:noop.reset();identity=vae.decode(z).sample
                finally:noop.close()
                r={'clip':clip,'edit':edit,'method':method,'prefix_max_vs_noop':float((y[:,:,:29]-identity[:,:,:29]).abs().max()),'prefix_max_vs_standard':float((y[:,:,:29]-base[:,:,:29]).abs().max()),'identity_max':float((identity-base).abs().max()),'raw_prefix_max':float((raw[:,:,:29]-base[:,:,:29]).abs().max()),'edit_relative_rmse':float((y[:,:,29:]-raw[:,:,29:]).square().mean().sqrt()/denom),'edit_amplitude_ratio':float((y[:,:,29:]-base[:,:,29:]).square().mean().sqrt()/denom)}
                save(r);videos[method]=y.cpu()
            torch.save({'videos':videos},OUT/f'{clip}_{edit}.pt')
    report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':main()
