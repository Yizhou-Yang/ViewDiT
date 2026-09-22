#!/usr/bin/env python3
"""Dual-reference normalization transport on a fixed Cog execution graph.

Preserve reference moments before a block boundary; use edited moments after it.
Not a standard decoder, and not a proof of dominance over pixel composition.
"""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLCogVideoX
from cog_norm_replay import ReplayNorm
ROOT = Path('/root/viewdit/data/boundary_holdout')
OUT = Path('/root/viewdit/results/video/cog_boundary_moment_transport')


class BoundaryMomentTransport:
    def __init__(self, model, reference, edited, target_call=1, mode='dual', ramp=0.):
        self.handles = []
        self.count = {}
        self.reference = reference
        self.edited = edited
        self.target_call = target_call
        self.mode = mode
        self.ramp = ramp
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.GroupNorm):
                self.handles.append(m.register_forward_hook(self.hook(name)))

    def hook(self, name):
        def apply(m, args, out):
            x = args[0]
            call = self.count.get(name, 0)
            self.count[name] = call + 1
            b, c, t, h, w = x.shape
            rm, rr = self.reference[(name, call)]
            em, er = self.edited[(name, call)]
            alpha = torch.zeros(t, device=x.device, dtype=x.dtype)
            if self.mode != 'reference':
                if call > self.target_call:
                    alpha[:] = 1
                elif call == self.target_call:
                    assert t % 2 == 0, (name, call, t)
                    boundary = t // 2
                    alpha[boundary:] = 1
                    if self.ramp:
                        alpha[boundary:] = torch.linspace(1/(t-boundary), 1, t-boundary, device=x.device) ** self.ramp
            shape = (b, m.num_groups, 1, 1, 1, 1)
            rm, rr, em, er = [a.reshape(shape) for a in [rm, rr, em, er]]
            alpha = alpha.reshape(1, 1, 1, t, 1, 1)
            gain = rr * (1-alpha) + er * alpha
            bias = -rm * rr * (1-alpha) - em * er * alpha
            y = (x.reshape(b, m.num_groups, c//m.num_groups, t, h, w) * gain + bias).reshape_as(x)
            if m.weight is not None:
                y = y * m.weight.view(1,c,1,1,1)
            if m.bias is not None:
                y = y + m.bias.view(1,c,1,1,1)
            return y
        return apply

    def reset(self):
        self.count = {}

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def capture(vae, z):
    norm = ReplayNorm(vae.decoder)
    try:
        norm.reset('capture')
        video = vae.decode(z).sample
        stats = {k: tuple(t.detach() for t in v) for k,v in norm.stats.items()}
    finally:
        norm.close()
    return video, stats


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    vae = AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE',local_files_only=True,torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report = {'complete':False,'protocol':{'clips':['bmx-trees','boat','bus','dog'],'size':[128,224],'block':4,'amplitudes':[.4,1.],'methods':['reference','dual','dual_ramp'],'warning':'Development set reused from previous round. Single last-block boundary with default decode batching. No semantic success claim. Pixel splice exact baseline retained.'},'rows':[]}
    def save(row=None):
        if row is not None:
            report['rows'].append(row)
            print('ROW',json.dumps(row),flush=True)
        p=OUT/'stats.tmp';p.write_text(json.dumps(report,indent=2));p.replace(OUT/'stats.json')
    for clip in report['protocol']['clips']:
        files=sorted((ROOT/'JPEGImages'/clip).glob('*.jpg'))[:17]
        x=torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((224,128)),dtype=np.float32)/127.5-1 for p in files])).permute(3,0,1,2).unsqueeze(0).cuda()
        z=vae.encode(x).latent_dist.mode()
        base, ref_stats=capture(vae,z)
        field=(torch.roll(z,1,-1)-torch.roll(z,-1,-1))*.5
        field=field/field.square().mean().sqrt()*z.square().mean().sqrt()
        for amplitude in report['protocol']['amplitudes']:
            ze=z.clone();ze[:,:,-1]+=amplitude*field[:,:,-1]
            raw, edit_stats=capture(vae,ze)
            denom=(raw[:,:,13:]-base[:,:,13:]).square().mean().sqrt().clamp(min=1e-12)
            videos={'base':base.cpu(),'raw':raw.cpu()}
            splice=raw.clone();splice[:,:,:13]=base[:,:,:13];videos['pixel_splice']=splice.cpu()
            for method in report['protocol']['methods']:
                control=BoundaryMomentTransport(vae.decoder,ref_stats,edit_stats,mode=method,ramp=1. if method=='dual_ramp' else 0.)
                try:
                    torch.cuda.synchronize();tic=time.perf_counter()
                    control.reset();y=vae.decode(ze).sample
                    torch.cuda.synchronize();seconds=time.perf_counter()-tic
                finally:
                    control.close()
                noop=BoundaryMomentTransport(vae.decoder,ref_stats,ref_stats,mode=method,ramp=1. if method=='dual_ramp' else 0.)
                try:
                    noop.reset();identity=vae.decode(z).sample
                finally:
                    noop.close()
                residual=y-base
                save({'clip':clip,'amplitude':amplitude,'method':method,'prefix_max_vs_base':float((y[:,:,:13]-base[:,:,:13]).abs().max()),'prefix_max_vs_noop':float((y[:,:,:13]-identity[:,:,:13]).abs().max()),'noop_max':float((identity-base).abs().max()),'edit_relative_rmse':float((y[:,:,13:]-raw[:,:,13:]).square().mean().sqrt()/denom),'edit_amplitude_ratio':float(residual[:,:,13:].square().mean().sqrt()/denom),'seam_residual_jump':float((residual[:,:,13]-residual[:,:,12]).square().mean().sqrt()),'seconds':seconds})
                videos[method]=y.cpu()
            torch.save({'videos':videos,'latent':ze.cpu()},OUT/f'{clip}_{amplitude}.pt')
    report['complete']=True;save();print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
