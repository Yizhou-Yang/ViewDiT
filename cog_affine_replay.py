#!/usr/bin/env python3
"""Compile reference GroupNorm moments to one affine kernel per invocation."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLCogVideoX
from cog_norm_replay import ReplayNorm
ROOT = Path('/root/viewdit/data/boundary_holdout')
OUT = Path('/root/viewdit/results/video/cog_affine_replay')


class CompiledReferenceNorm:
    """Scoped forward replacement. Reset invocation counts before EVERY decode.

    Stored coefficients are tied to reference video, shape, and execution batching.
    This is NOT an unmodified decoder or a learned adapter.
    """
    def __init__(self, model, moments):
        self.originals = []
        self.counts = {}
        self.coefficients = {}
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.GroupNorm):
                continue
            self.originals.append((module, module.forward))
            for (key, call), (mu, rs) in moments.items():
                if key != name:
                    continue
                channels = module.num_channels
                gain = rs.repeat_interleave(channels // module.num_groups, dim=1).reshape(-1, channels, 1, 1, 1)
                mean = mu.repeat_interleave(channels // module.num_groups, dim=1).reshape_as(gain)
                if module.weight is not None:
                    gain = gain * module.weight.view(1, channels, 1, 1, 1)
                bias = -mean * gain
                if module.bias is not None:
                    bias = bias + module.bias.view(1, channels, 1, 1, 1)
                self.coefficients[(name, call)] = (gain.detach(), bias.detach())
            module.forward = self._forward(name)

    def _forward(self, name):
        def forward(x):
            call = self.counts.get(name, 0)
            self.counts[name] = call + 1
            gain, bias = self.coefficients[(name, call)]
            return torch.addcmul(bias, x, gain)
        return forward

    def reset(self):
        self.counts = {}

    def close(self):
        for module, original in self.originals:
            module.forward = original
        self.originals = []


def timed_decode(vae, z, reset=None, repetitions=3):
    samples = []
    result = None
    for _ in range(repetitions):
        if reset:
            reset()
        torch.cuda.synchronize()
        tic = time.perf_counter()
        result = vae.decode(z).sample
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - tic)
    return result, sorted(samples)[len(samples) // 2]


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE', local_files_only=True, torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report = {'complete': False, 'protocol': {'clips': ['bmx-trees', 'boat', 'bus', 'dog'], 'resolution': [128, 224], 'frames': 17, 'blocks': [2, 4], 'amplitude': .4, 'benchmark': 'Median of3 synchronized decodes after warmup. Capture and compilation reported separately. Raw/hook/compiled decoded outputs retain same shape and batching.', 'warning': 'Compilation uses same held-out content as mechanism test; an implementation audit, not new independent validation. Reference moments and execution config are required side information.'}, 'rows': []}
    def save(row=None):
        if row is not None:
            report['rows'].append(row)
            print('ROW', json.dumps(row), flush=True)
        p = OUT / 'stats.tmp'
        p.write_text(json.dumps(report, indent=2))
        p.replace(OUT / 'stats.json')
    for clip in report['protocol']['clips']:
        files = sorted((ROOT / 'JPEGImages' / clip).glob('*.jpg'))[:17]
        arrays = [np.asarray(Image.open(p).convert('RGB').resize((224, 128)), dtype=np.float32) / 127.5 - 1 for p in files]
        x = torch.from_numpy(np.stack(arrays)).permute(3, 0, 1, 2).unsqueeze(0).cuda()
        z = vae.encode(x).latent_dist.mode()
        base = vae.decode(z).sample
        field = (torch.roll(z, 1, -1) - torch.roll(z, -1, -1)) * .5
        field = field / field.square().mean().sqrt() * z.square().mean().sqrt()
        for block in [2, 4]:
            start = 1 + 4 * (block - 1)
            ze = z.clone()
            ze[:, :, block] += .4 * field[:, :, block]
            raw, raw_sec = timed_decode(vae, ze)
            norm = ReplayNorm(vae.decoder)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            norm.reset('capture')
            vae.decode(z)
            torch.cuda.synchronize()
            capture_sec = time.perf_counter() - tic
            moments = norm.stats
            hooked, hook_sec = timed_decode(vae, ze, lambda: norm.reset('replay'))
            norm.close()
            torch.cuda.synchronize()
            tic = time.perf_counter()
            compiled = CompiledReferenceNorm(vae.decoder, moments)
            torch.cuda.synchronize()
            compile_sec = time.perf_counter() - tic
            compiled.reset()
            noop = vae.decode(z).sample
            final, affine_sec = timed_decode(vae, ze, compiled.reset)
            runtime_scalars = sum(a.numel() + b.numel() for a, b in compiled.coefficients.values())
            compiled.close()
            restored = vae.decode(z).sample
            assert float((restored - base).abs().max()) == 0
            edit_rms = (raw[:, :, start:start+4] - base[:, :, start:start+4]).square().mean().sqrt()
            save({'clip': clip, 'block': block, 'raw_seconds': raw_sec, 'hook_seconds': hook_sec, 'affine_seconds': affine_sec, 'capture_seconds': capture_sec, 'compile_seconds': compile_sec, 'affine_vs_raw_latency_ratio': affine_sec / raw_sec, 'affine_prefix_max': float((final[:, :, :start] - noop[:, :, :start]).abs().max()), 'affine_noop_max': float((noop - base).abs().max()), 'affine_vs_hook_max': float((final - hooked).abs().max()), 'edit_relative_rmse_vs_raw': float((final[:, :, start:start+4] - raw[:, :, start:start+4]).square().mean().sqrt() / edit_rms), 'moment_scalars': sum(a.numel() + b.numel() for a, b in moments.values()), 'runtime_coefficient_scalars': runtime_scalars, 'latent_scalars': z.numel(), 'raw_prefix_max': float((raw[:, :, :start] - base[:, :, :start]).abs().max())})
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
