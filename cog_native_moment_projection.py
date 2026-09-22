#!/usr/bin/env python3
"""Pilot native-latent projection: moment constraints vs direct pixel fitting."""
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
ROOT = Path('/root/viewdit/results/video/wan_real_pilot')
OUT = Path('/root/viewdit/results/video/cog_native_moment_projection')


class MomentLoss:
    def __init__(self, model):
        self.mode = 'capture'
        self.reference = {}
        self.count = {}
        self.terms = []
        self.handles = [m.register_forward_hook(self.hook(n)) for n, m in model.named_modules() if isinstance(m, torch.nn.GroupNorm)]

    def hook(self, name):
        def apply(m, args, out):
            x = args[0]
            i = self.count.get(name, 0)
            self.count[name] = i + 1
            if x.ndim != 5:
                return out
            v = x.reshape(x.shape[0], m.num_groups, -1)
            mu = v.mean(-1)
            std = (v.var(-1, unbiased=False) + m.eps).sqrt()
            if self.mode == 'capture':
                self.reference[(name, i)] = (mu.detach(), std.detach())
            else:
                rm, rs = self.reference[(name, i)]
                self.terms.append(((mu-rm)/rs).square().mean() + (std/rs-1).square().mean())
            return out
        return apply

    def reset(self, mode):
        self.mode = mode
        self.count = {}
        self.terms = []

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE', local_files_only=True, torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report = {'complete': False, 'protocol': {'clips': ['bear', 'blackswan'], 'resolution': [64, 96], 'block': 4, 'amplitude': .4, 'steps': 32, 'lr': .01, 'lambdas': [1., 10.], 'methods': ['moment', 'pixel'], 'warning': 'Exploratory native projection pilot, no certification. Direct raw decoded edit is the fidelity target. Same backward count, measured time, all candidates retained.'}, 'rows': []}
    def save(row=None):
        if row is not None:
            report['rows'].append(row)
            print('ROW', json.dumps(row), flush=True)
        p = OUT / 'stats.tmp'
        p.write_text(json.dumps(report, indent=2))
        p.replace(OUT / 'stats.json')
    for clip in report['protocol']['clips']:
        d = torch.load(ROOT / (clip+'_latent_geometric_impulse.pt'), map_location='cpu', weights_only=True)
        v = d['videos']['base'].cuda()
        x = F.interpolate(v.permute(1, 0, 2, 3), size=(64, 96), mode='bilinear', align_corners=False).permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
            base = vae.decode(z).sample
            field = (torch.roll(z, 1, -1)-torch.roll(z, -1, -1))*.5
            field = field/field.square().mean().sqrt()*z.square().mean().sqrt()
            ze = z.clone()
            ze[:, :, -1] += .4*field[:, :, -1]
            raw = vae.decode(ze).sample
            prefix_initial = (raw[:, :, 9:13]-base[:, :, 9:13]).square().mean().clamp(min=1e-12)
            edit_energy = (raw[:, :, 13:]-base[:, :, 13:]).square().mean().clamp(min=1e-12)
        for method in report['protocol']['methods']:
            for weight in report['protocol']['lambdas']:
                hooks = None
                if method == 'moment':
                    hooks = MomentLoss(vae.decoder)
                    with torch.no_grad():
                        hooks.reset('capture')
                        vae.decode(z)
                        hooks.reset('measure')
                        vae.decode(ze)
                        moment_initial = torch.stack(hooks.terms).mean().clamp(min=1e-12).detach()
                delta = torch.zeros_like(ze[:, :, -1:], requires_grad=True)
                opt = torch.optim.Adam([delta], lr=.01)
                tic = time.time()
                for _ in range(32):
                    opt.zero_grad(set_to_none=True)
                    zz = torch.cat([ze[:, :, :-1], ze[:, :, -1:]+delta], dim=2)
                    if hooks:
                        hooks.reset('measure')
                    y = vae.decode(zz).sample
                    protected_loss = torch.stack(hooks.terms).mean()/moment_initial if hooks else (y[:, :, 9:13]-base[:, :, 9:13]).square().mean()/prefix_initial
                    edit_loss = (y[:, :, 13:]-raw[:, :, 13:]).square().mean()/edit_energy
                    loss = protected_loss + weight*edit_loss + .001*delta.square().mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                if hooks:
                    hooks.close()
                with torch.no_grad():
                    zz = torch.cat([ze[:, :, :-1], ze[:, :, -1:]+delta], dim=2)
                    final = vae.decode(zz).sample
                    mse = (final[:, :, 9:13]-base[:, :, 9:13]).square().mean()
                    save({'clip': clip, 'method': method, 'edit_weight': weight, 'prefix_mse_reduction_pct': float(100*(1-mse/prefix_initial)), 'prefix_rms': float(mse.sqrt()), 'prefix_max': float((final[:, :, :13]-base[:, :, :13]).abs().max()), 'raw_prefix_max': float((raw[:, :, :13]-base[:, :, :13]).abs().max()), 'edit_relative_rmse': float(((final[:, :, 13:]-raw[:, :, 13:]).square().mean()/edit_energy).sqrt()), 'seconds': time.time()-tic, 'backwards': 32, 'standard_decoder': True})
                    torch.save({'latent': zz.cpu(), 'videos': {'base': base.cpu(), 'raw': raw.cpu(), 'projected': final.cpu()}}, OUT / f'{clip}_{method}_{weight}.pt')
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
