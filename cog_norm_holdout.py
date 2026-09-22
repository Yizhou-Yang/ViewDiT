#!/usr/bin/env python3
"""Frozen-protocol cross-content validation of reference-moment replay."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLCogVideoX
from cog_norm_replay import ReplayNorm
ROOT = Path('/root/viewdit/data/boundary_holdout')
OUT = Path('/root/viewdit/results/video/cog_norm_holdout')


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256']
    vae = AutoencoderKLCogVideoX.from_pretrained('/root/viewdit/weights/CogVAE', local_files_only=True, torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    report = {'complete': False, 'protocol': {'clips': manifest['clips'], 'height': 128, 'width': 224, 'frames': 17, 'latent_blocks': [2, 4], 'edit_types': ['geometric_0.4', 'encoded_color_patch'], 'float': 'fp32', 'normalization': 'reference moments frozen for all decoder GroupNorm calls; no fitting', 'criteria': {'prefix_max_vs_replay_noop': 1e-6, 'noop_max_vs_standard': 1e-4, 'edited_block_relative_rmse_vs_raw': .1}, 'warning': 'Direct new DAVIS content, no Wan preprocessing. Mechanism validation, not semantic editor/DiT reuse/perceptual benchmark. Pixel splice remains an exact rendered baseline.'}, 'rows': []}
    def save(row=None):
        if row is not None:
            report['rows'].append(row)
            print('ROW', json.dumps(row), flush=True)
        p = OUT / 'stats.tmp'
        p.write_text(json.dumps(report, indent=2))
        p.replace(OUT / 'stats.json')
    for clip in manifest['clips']:
        files = sorted((ROOT / 'JPEGImages' / clip).glob('*.jpg'))[:17]
        arrays = [np.asarray(Image.open(p).convert('RGB').resize((224, 128)), dtype=np.float32) / 127.5 - 1 for p in files]
        x = torch.from_numpy(np.stack(arrays)).permute(3, 0, 1, 2).unsqueeze(0).cuda()
        z = vae.encode(x).latent_dist.mode()
        original = vae.decode(z).sample
        field = (torch.roll(z, 1, -1) - torch.roll(z, -1, -1)) * .5
        field = field / field.square().mean().sqrt() * z.square().mean().sqrt()
        for block in [2, 4]:
            start = 1 + 4 * (block - 1)
            end = start + 4
            px = x.clone()
            px[:, 0, start:end, 32:96, 56:168] = (px[:, 0, start:end, 32:96, 56:168] + .35).clamp(-1, 1)
            color_z = vae.encode(px).latent_dist.mode()
            for kind in report['protocol']['edit_types']:
                ze = z.clone()
                ze[:, :, block] = z[:, :, block] + .4 * field[:, :, block] if kind == 'geometric_0.4' else color_z[:, :, block]
                torch.cuda.synchronize()
                tic = time.time()
                raw = vae.decode(ze).sample
                torch.cuda.synchronize()
                raw_sec = time.time() - tic
                norm = ReplayNorm(vae.decoder)
                norm.reset('capture')
                tic = time.time()
                ref = vae.decode(z).sample
                torch.cuda.synchronize()
                capture_sec = time.time() - tic
                assert float((ref - original).abs().max()) == 0
                norm.reset('replay')
                replayed = vae.decode(z).sample
                norm.reset('replay')
                tic = time.time()
                fixed = vae.decode(ze).sample
                torch.cuda.synchronize()
                replay_sec = time.time() - tic
                moment_scalars = sum(a.numel() + b.numel() for a, b in norm.stats.values())
                norm.reset('framewise')
                fw_ref = vae.decode(z).sample
                norm.reset('framewise')
                fw_edit = vae.decode(ze).sample
                norm.close()
                denom = (raw[:, :, start:end] - original[:, :, start:end]).square().mean().sqrt().clamp(min=1e-12)
                prefix_max = float((fixed[:, :, :start] - replayed[:, :, :start]).abs().max())
                identity_max = float((replayed - original).abs().max())
                distortion = float((fixed[:, :, start:end] - raw[:, :, start:end]).square().mean().sqrt() / denom)
                pixel_splice = raw.clone()
                pixel_splice[:, :, :start] = original[:, :, :start]
                r = {'clip': clip, 'block': block, 'edit': kind, 'prefix_frames': start, 'raw_prefix_rms': float((raw[:, :, :start] - original[:, :, :start]).square().mean().sqrt()), 'raw_prefix_max': float((raw[:, :, :start] - original[:, :, :start]).abs().max()), 'replay_prefix_rms': float((fixed[:, :, :start] - replayed[:, :, :start]).square().mean().sqrt()), 'replay_prefix_max': prefix_max, 'identity_max': identity_max, 'identity_mse': float((replayed - original).square().mean()), 'edit_relative_rmse_vs_raw': distortion, 'edit_amplitude_ratio': float((fixed[:, :, start:end] - replayed[:, :, start:end]).square().mean().sqrt() / denom), 'framewise_noop_mse': float((fw_ref - original).square().mean()), 'framewise_prefix_max': float((fw_edit[:, :, :start] - fw_ref[:, :, :start]).abs().max()), 'moment_scalars': moment_scalars, 'latent_scalars': z.numel(), 'capture_seconds': capture_sec, 'raw_decode_seconds': raw_sec, 'replay_decode_seconds': replay_sec, 'passes_all': prefix_max <= 1e-6 and identity_max <= 1e-4 and distortion <= .1}
                save(r)
                torch.save({'videos': {'base': original.cpu(), 'raw': raw.cpu(), 'replay': fixed.cpu(), 'pixel_splice': pixel_splice.cpu(), 'framewise': fw_edit.cpu()}}, OUT / f'{clip}_{block}_{kind}.pt')
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
