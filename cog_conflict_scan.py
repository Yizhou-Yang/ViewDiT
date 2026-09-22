#!/usr/bin/env python3
"""Conflict control scan on real CogVideoX generated latent (OOM-safe).

Scan a continuous protection strength alpha over the decoder GroupNorm calls
that we have reference+edited statistics for. Calls not covered by the
captured statistics are passed through unchanged (no intervention). All
comparison videos stay on CPU; per-iteration decode is freed immediately.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time
from pathlib import Path
import torch
from diffusers import AutoencoderKLCogVideoX
from cog_boundary_moment_transport import capture
from cog_exact_memory import NoUnusedTemporalCache

SRC = Path('/root/viewdit/results/video/cog_semantic_bear49')
OUT = Path('/root/viewdit/results/video/cog_conflict_scan')


class AlphaScanNorm:
    """Blend reference/edited norm stats on covered GroupNorm calls.

    reference/edited: {(name, call): (mu, invrms)} from capture().
    For each covered call, the SECOND temporal half (edited future) is scaled
    by `alpha` toward edited moments; the first half stays on reference
    moments (protection). Calls absent from either dict pass through.
    """
    def __init__(self, model, reference, edited, alpha=0.0):
        self.count = {}
        self.handles = []
        self.reference = reference
        self.edited = edited
        self.alpha = alpha
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.GroupNorm):
                self.handles.append(m.register_forward_hook(self.hook(name)))

    def hook(self, name):
        def apply(m, args, out):
            call = self.count.get(name, 0)
            self.count[name] = call + 1
            key = (name, call)
            if key not in self.reference or key not in self.edited:
                return out  # not covered by captured stats: pass through
            rm, rr = self.reference[key]
            em, er = self.edited[key]
            x = args[0]
            b, c, t, h, w = x.shape
            alpha = torch.zeros(t, device=x.device, dtype=x.dtype)
            if t % 2 == 0:
                alpha[t // 2:] = self.alpha
            else:
                alpha[:] = self.alpha
            shape = (b, m.num_groups, 1, 1, 1, 1)
            rm, rr, em, er = [a.reshape(shape) for a in [rm, rr, em, er]]
            alpha = alpha.reshape(1, 1, 1, t, 1, 1)
            gain = rr * (1 - alpha) + er * alpha
            bias = -rm * rr * (1 - alpha) - em * er * alpha
            y = (x.reshape(b, m.num_groups, c // m.num_groups, t, h, w) * gain + bias).reshape_as(x)
            if m.weight is not None:
                y = y * m.weight.view(1, c, 1, 1, 1)
            if m.bias is not None:
                y = y + m.bias.view(1, c, 1, 1, 1)
            return y
        return apply

    def reset(self):
        self.count = {}

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        '/root/viewdit/weights/CogVAE', local_files_only=True,
        torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    patch = NoUnusedTemporalCache(vae)
    report = {'complete': False,
              'protocol': {
                  'source': 'real 49-frame CogVideoX generated edited latent (bear_round1)',
                  'reference': 'full original latent decode stats',
                  'edited': 'full edited latent decode stats (freed after capture)',
                  'boundary_rule': 'each covered GroupNorm call: first temporal half protected, second half scaled by alpha',
                  'protected_frames': 13,
                  'scan': 'alpha in {0,.05,.1,.2,.3,.5,.7,.85,1.0}',
                  'metrics': 'protected_prefix_max vs base; edited_relative_rmse_vs_raw; edited_amplitude_ratio',
                  'warning': 'Decoder-side conflict scan on one generated latent. Not an edit-success result.',
              }, 'rows': []}

    def save(row=None):
        if row is not None:
            report['rows'].append(row)
            print('ROW', json.dumps(row), flush=True)
        p = OUT / 'stats.tmp'
        p.write_text(json.dumps(report, indent=2))
        p.replace(OUT / 'stats.json')

    d = torch.load(SRC / 'bear_round1.pt', map_location='cpu', weights_only=True)
    z = d['source_latent'].float()
    ze = d['latent'].float()
    base = d['videos']['base']
    raw = d['videos']['raw']
    del d
    torch.cuda.empty_cache()

    _, ref_stats = capture(vae, z.cuda())
    torch.cuda.empty_cache()
    _, edit_stats = capture(vae, ze.cuda())
    torch.cuda.empty_cache()
    ze_gpu = ze.cuda()

    protected_end = 13
    denom = (raw[:, :, protected_end:] - base[:, :, protected_end:]).square().mean().sqrt().clamp(min=1e-12)

    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]
    results = {}
    start = time.time()
    for alpha in alphas:
        control = AlphaScanNorm(vae.decoder, ref_stats, edit_stats, alpha=alpha)
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = vae.decode(ze_gpu).sample.float().cpu()
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0
        finally:
            control.close()
        del control
        torch.cuda.empty_cache()
        protected_max = float((y[:, :, :protected_end] - base[:, :, :protected_end]).abs().max())
        edit_rel = float((y[:, :, protected_end:] - raw[:, :, protected_end:]).square().mean().sqrt() / denom)
        edit_amp = float((y[:, :, protected_end:] - base[:, :, protected_end:]).square().mean().sqrt() / denom)
        results[alpha] = {'alpha': alpha, 'protected_prefix_max': protected_max,
                          'edited_relative_rmse_vs_raw': edit_rel,
                          'edited_amplitude_ratio': edit_amp, 'decode_seconds': secs}
        save(results[alpha])
        if alpha in (0.0, 0.5, 1.0):
            torch.save({'video': y}, OUT / f'alpha_{alpha}.pt')
        del y
        torch.cuda.empty_cache()
    report['rows'] = [results[a] for a in alphas]
    report['wall_seconds'] = time.time() - start
    patch.close()
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()