#!/usr/bin/env python3
"""Causal audit: matched DDPM noise, true spectral edits, collapse controls.
Uses existing ViewDiT environment. No model downloads, no training.
LATTE_N controls seeds; outputs incremental rows and contact sheets.
"""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_xl_gate import load_model, _noise, _forward, one_step, x0_from_eps, eps_from_x0, temporal_smooth
from tis3 import _dct_mat, gate_high_freq as old_gate
from tis_probe import frame_lowpass as old_lowpass
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid, save_image

OUT = Path(os.environ.get('VIEWDIT_VIDEO_OUT', '/root/viewdit/results/video/tis_review'))
N = int(os.environ.get('LATTE_N', '2'))


def spectral(x, retention, cut=8):
    if retention == 1:
        return x
    B, F, C, H, W = x.shape
    y = x.permute(0, 2, 3, 4, 1).reshape(-1, F).float()
    d = _dct_mat(F, x.device, torch.float32)
    c = y @ d.T
    c[:, max(1, F-cut):] *= retention
    return (c @ d).reshape(B, C, H, W, F).permute(0, 4, 1, 2, 3).to(x.dtype)


def cos_adj(x):
    y = x[0].float().flatten(1)
    return float(torch.nn.functional.cosine_similarity(y[:-1], y[1:], dim=1).mean())


def structure(x):
    centered = x - x.mean(dim=1, keepdim=True)
    return {'raw_cos': cos_adj(x), 'temporal_centered_cos': cos_adj(centered),
            'temporal_residual_energy': float(centered.float().square().sum() / (x.float().square().sum()+1e-12))}


@torch.no_grad()
def sample(model, diffusion, z, seed, mode, trace=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = z.clone()
    wrapped = diffusion._wrap_model(model)
    a = torch.as_tensor(diffusion.alphas_cumprod, device=z.device, dtype=torch.float32)
    records = []
    start = time.time()
    for i, idx in enumerate(reversed(range(diffusion.num_timesteps))):
        t = torch.full((1,), idx, device=z.device, dtype=torch.long)
        out = _forward(wrapped, x, t)
        eps = out[:, :, :4]
        alpha = a[idx].view(1, 1, 1, 1, 1)
        x0 = x0_from_eps(x, eps, alpha)
        if trace and i in [0, 25, 75, 125, 175, 225, 249]:
            records.append({'iteration': i, 'alpha_bar': float(a[idx]), 'eps': structure(eps), 'x0': structure(x0)})
        edit = None
        if i >= int(0.7 * diffusion.num_timesteps):
            if mode == 'dct_weak':
                edit = spectral(x0, 0.85)
            elif mode == 'dct_zero':
                edit = spectral(x0, 0.0)
            elif mode == 'smooth':
                edit = temporal_smooth(x0, 0.25)
            elif mode == 'identity':
                edit = spectral(x0, 1.0)
        if edit is not None and mode != 'identity':
            out = torch.cat([eps_from_x0(x, edit, alpha), out[:, :, 4:]], dim=2)
        x = one_step(diffusion, out, x, t)
    torch.cuda.synchronize()
    return x, records, time.time()-start


@torch.no_grad()
def decode(vae, x):
    lat = x[0].float()/0.18215
    return torch.cat([vae.decode(lat[i:i+4]).sample.clamp(-1, 1) for i in range(0, len(lat), 4)]).cpu()


def metrics(img, ref):
    x, r = img.float(), ref.float()
    mse = float((x-r).square().mean())
    d1 = x[1:]-x[:-1]
    d2 = x[2:]-2*x[1:-1]+x[:-2]
    gx, gy = x[:, :, :, 1:]-x[:, :, :, :-1], x[:, :, 1:, :]-x[:, :, :-1, :]
    return {'psnr_vs_full': 120.0 if mse == 0 else float(10*np.log10(4/mse)),
            'pixel_l1_vs_full': float((x-r).abs().mean()),
            'frame_diff_proxy': float(d1.flatten(1).norm(dim=1).mean()/(x[:-1].flatten(1).norm(dim=1).mean()+1e-8)),
            'temporal_second_diff': float(d2.abs().mean()),
            'spatial_gradient': float((gx.abs().mean()+gy.abs().mean())/2),
            'temporal_std': float(x.std(dim=0).mean())}


def save_report(report):
    temp = OUT/'stats.tmp'
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(OUT/'stats.json')


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    torch.manual_seed(42)
    probe = torch.randn(1, 16, 4, 8, 8, device=device)
    d = _dct_mat(16, device, torch.float32)
    ramp = torch.arange(16, device=device).float().view(1,16,1,1,1).expand(1,16,4,8,8).contiguous()
    unit = {'old_alpha0_is_noop': bool(torch.equal(old_gate(probe,0,8), probe)),
            'corrected_alpha0_changes_tensor': bool(not torch.equal(spectral(probe,0), probe)),
            'identity_exact': bool(torch.equal(spectral(probe,1),probe)),
            'dct_orthogonal_max_error': float((d@d.T-torch.eye(16,device=device)).abs().max()),
            'old_frame_lowpass_ramp_interior_max_error': float((old_lowpass(ramp,3)[:,1:-1]-ramp[:,1:-1]).abs().max())}
    print('UNIT', json.dumps(unit), flush=True)
    model, diffusion = load_model(device)
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema', local_files_only=True).to(device).eval()
    report = {'protocol': {'N':N, 'T':diffusion.num_timesteps, 'model':'Latte-XL/2 FFS', 'noise':'reset global CPU/CUDA RNG for each paired trajectory', 'late_fraction':0.7, 'metric_caution':'frame difference and temporal std are NOT motion quality; zero-motion control provided'}, 'unit_tests':unit, 'rows':[], 'structure':[], 'complete':False}
    save_report(report)
    for s in range(N):
        z = _noise(device,1000+s)
        ref_x, trace, secs = sample(model,diffusion,z,9000+s,'full',True)
        ref = decode(vae,ref_x)
        report['structure'].append({'seed':s,'trace':trace})
        vids = [ref]
        names = ['full']
        baseline = metrics(ref,ref)
        baseline.update(seed=s, mode='full', seconds=secs, latent_relL2=0.0)
        report['rows'].append(baseline)
        save_report(report)
        print('ROW',json.dumps(baseline),flush=True)
        for mode in ['identity','independent_noise','dct_weak','dct_zero','smooth']:
            noise_seed = 19000+s if mode == 'independent_noise' else 9000+s
            x, _, secs = sample(model,diffusion,z,noise_seed,mode)
            img = decode(vae,x)
            row = metrics(img,ref)
            row.update(seed=s,mode=mode,seconds=secs,latent_relL2=float((x-ref_x).float().norm()/(ref_x.float().norm()+1e-12)))
            report['rows'].append(row)
            vids.append(img)
            names.append(mode)
            save_report(report)
            print('ROW',json.dumps(row),flush=True)
        for mode,img in [('pixel_post_smooth', temporal_smooth(ref.unsqueeze(0),0.25)[0]),('frozen_frame',ref[:1].expand_as(ref))]:
            row = metrics(img,ref)
            row.update(seed=s,mode=mode,seconds=0)
            report['rows'].append(row)
            vids.append(img)
            names.append(mode)
        save_image(make_grid(torch.cat([(v[[0,4,8,12]]+1)/2 for v in vids]),nrow=4),OUT/f'seed{s}_rows.png')
        (OUT/f'seed{s}_rows.txt').write_text('\n'.join(names))
        torch.save({'names':names,'videos':torch.stack(vids)},OUT/f'seed{s}_videos.pt')
        save_report(report)
    report['complete']=True
    save_report(report)
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    main()
