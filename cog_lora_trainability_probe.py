#!/usr/bin/env python3
"""Resource/gradient probe only; NOT instruction editing or quality evidence.
Requires existing torch 2.6.0, diffusers 0.31.0 and transformers 4.45.2.
No PEFT install, no base checkpoint writes, no pretrained weights downloaded.
"""
import argparse
import json
import math
import time
import traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler


class ProbeLoRA(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        with torch.autocast('cuda', enabled=False):
            delta = F.linear(F.linear(x.float(), self.a), self.b)
        return y + delta.to(y.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('/root/viewdit/weights/CogVideoX-2b'))
    p.add_argument('--cache', type=Path, default=Path('/root/viewdit/results/video/cog_semantic_bear49_strong'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--height', type=int, default=128)
    p.add_argument('--width', type=int, default=224)
    p.add_argument('--frames', type=int, default=17)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--rank', type=int, default=8)
    args = p.parse_args()
    assert args.height % 16 == 0 and args.width % 16 == 0
    assert (args.frames - 1) % 4 == 0 and args.frames <= 49
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {'complete': False, 'purpose': 'trainability only; cached source self-denoising, not paired instruction editing', 'protocol': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'rows': []}

    def save():
        tmp = args.out / 'stats.tmp'
        tmp.write_text(json.dumps(stats, indent=2))
        tmp.replace(args.out / 'stats.json')

    save()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(20260922)
        stats['torch'] = torch.__version__
        stats['gpu'] = torch.cuda.get_device_name(0)
        model = CogVideoXTransformer3DModel.from_pretrained(args.model / 'transformer', torch_dtype=torch.float16, local_files_only=True)
        model.requires_grad_(False)
        targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and (n.endswith('attn1.to_q') or n.endswith('attn1.to_v'))]
        assert len(targets) == 60, len(targets)
        for name, module in targets:
            parent_name, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), attr, ProbeLoRA(module, args.rank))
        model.cuda().train()
        model.enable_gradient_checkpointing()
        params = [p for p in model.parameters() if p.requires_grad]
        stats['trainable_parameters'] = sum(p.numel() for p in params)
        stats['total_parameters'] = sum(p.numel() for p in model.parameters())
        stats['adapter_dtype'] = str(params[0].dtype)
        stats['base_dtype'] = str(model.patch_embed.proj.weight.dtype)
        stats['adapter_targets'] = [n for n, _ in targets]
        opt = torch.optim.AdamW(params, lr=1e-4)
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        cached = torch.load(args.cache / 'bear_round1.pt', map_location='cpu', weights_only=True)
        z = cached['source_latent'][:, :, :(args.frames - 1) // 4 + 1].float()
        del cached
        stats['latent_spatial_resampled'] = list(z.shape[-2:]) != [args.height // 8, args.width // 8]
        if stats['latent_spatial_resampled']:
            z = F.interpolate(z, size=(z.shape[2], args.height // 8, args.width // 8), mode='trilinear', align_corners=False)
        vae_config = json.loads((args.model / 'vae' / 'config.json').read_text())
        z = (z * vae_config['scaling_factor']).permute(0, 2, 1, 3, 4).contiguous().cuda().half()
        embeddings = torch.load(args.cache / 'embeddings.pt', map_location='cpu', weights_only=True)
        text = embeddings[''].cuda().half()
        del embeddings
        scheduler = CogVideoXDDIMScheduler.from_pretrained(args.model / 'scheduler', local_files_only=True)
        stats['prediction_type'] = scheduler.config.prediction_type
        stats['latent_shape'] = list(z.shape)
        noise = torch.randn_like(z)
        t = torch.tensor([500], device='cuda', dtype=torch.long)
        noisy = scheduler.add_noise(z, noise, t)
        if scheduler.config.prediction_type == 'v_prediction':
            target = scheduler.get_velocity(z, noise, t)
        elif scheduler.config.prediction_type == 'epsilon':
            target = noise
        else:
            raise ValueError(scheduler.config.prediction_type)
        stats['note'] = 'Fixed noise/time/batch, empty text, no source-conditioning adapter, no VAE/T5 on GPU. FP32 LoRA A/B, frozen FP16 base, non-reentrant checkpointing. Timers exclude model loading and data preparation.'
        save()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                pred = model(hidden_states=noisy, encoder_hidden_states=text, timestep=t, return_dict=False)[0]
                loss = F.mse_loss(pred.float(), target.float())
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            before = params[1].detach().clone()
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            update_max = float((params[1] - before).abs().max())
            del before
            torch.cuda.synchronize()
            row = {'step': step, 'loss': float(loss.detach()), 'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'finite_gradients': finite, 'gradient_norm_before_clip': grad_norm, 'first_B_update_max': update_max, 'scale_before': scale_before, 'scale_after': scaler.get_scale()}
            stats['rows'].append(row)
            print(json.dumps(row), flush=True)
            save()
            if not finite or update_max == 0:
                raise RuntimeError('invalid gradients or optimizer skipped update')
            del loss, pred
        stats['complete'] = True
        save()
    except Exception as exc:
        stats['error'] = repr(exc)
        stats['traceback'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
