#!/usr/bin/env python3
"""Held-out synthetic localization validation; not a semantic benchmark."""
import json
import time
from pathlib import Path
import torch
import endpoint_control as ec
from run_xl_gate import load_model, _forward, one_step
from tis_review import decode
from diffusers.models import AutoencoderKL
from PIL import Image, ImageDraw

OUT = Path('/root/viewdit/results/video/endpoint_validate')
SEEDS = [50, 51, 52, 53]
BUDGET = 8


@torch.no_grad()
def rollout(model, diff, states, rng, delta, active):
    torch.cuda.set_rng_state(rng)
    x = states[0] + delta
    wrapped = diff._wrap_model(model)
    for j, idx in enumerate(reversed(range(diff.num_timesteps - ec.START))):
        t = torch.full((1,), idx, device=x.device, dtype=torch.long)
        x = one_step(diff, _forward(wrapped, x, t), x, t)
        x[:, ~active] = states[j + 1][:, ~active]
    return x.float()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    model, diff = load_model(torch.device('cuda'))
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema', local_files_only=True).to('cuda').eval()
    report = {'protocol': {'seeds': SEEDS, 'extra_tail_budget': BUDGET, 'tail_steps': diff.num_timesteps - ec.START, 'amplitude': 0.2, 'task': 'synthetic native edit localization; two unseen envelope shapes', 'selection': 'fixed final iterate versus online accepted iterate; all rejected evaluations counted', 'safeguard': 'Anderson history 3, ridge 0.001, l1 coefficient bound 4, eta 0.8 halved on rejection to minimum 0.05'}, 'rows': [], 'complete': False}
    def save():
        (OUT / 'stats.json').write_text(json.dumps(report, indent=2))
    for seed in SEEDS:
        states, rng = ec.base_run(model, diff, seed)
        base = states[-1].float()
        field = ec.field_of(base)
        env = torch.zeros(16, device='cuda')
        if seed % 2 == 0:
            env[4:12] = torch.tensor([0.25, 0.5, 0.75, 1., 1., 0.75, 0.5, 0.25], device='cuda')
            curve = 'wide_ramp'
        else:
            env[3:6] = torch.tensor([0.5, 1., 0.5], device='cuda')
            env[10:13] = torch.tensor([0.5, 1., 0.5], device='cuda')
            curve = 'two_pulses'
        active = env > 0
        mask = active.view(1, 16, 1, 1, 1)
        native = ec.rollout(model, diff, states, rng, field, torch.ones(16, device='cuda') * 0.2)
        target = base + (native - base) * env.view(1, 16, 1, 1, 1)
        initial = field * (0.2 * env).view(1, 16, 1, 1, 1)
        replay = rollout(model, diff, states, rng, torch.zeros_like(initial), active)
        assert float((replay - base).abs().max()) == 0
        denom = (target - base).norm().clamp(min=1e-8)
        def loss(y):
            return float(((target - y) * mask).norm() / denom)
        base_img, target_img = decode(vae, base), decode(vae, target)
        videos = {'base': base_img, 'target_composite': target_img}
        def record(name, y, extra):
            img = decode(vae, y)
            desired = target - base
            row = {'seed': seed, 'curve': curve, 'method': name, 'target_rel_error': loss(y), 'achieved_target_gain': float(((y - base) * desired).sum() / desired.square().sum()), 'outside_max_latent_error': float((y[:, ~active] - base[:, ~active]).abs().max()), 'pixel_target_mse': float((img[active.cpu()] - target_img[active.cpu()]).square().mean()), 'pixel_outside_max_error': float((img[~active.cpu()] - base_img[~active.cpu()]).abs().max()), 'target_pixel_rms': float((target_img - base_img).square().mean().sqrt())}
            row.update(extra)
            report['rows'].append(row)
            videos[name] = img
            print('ROW', json.dumps(row), flush=True)
            save()
        y0 = rollout(model, diff, states, rng, initial, active)
        record('hard_direct', y0, {'extra_nfe': 0})
        start = time.time()
        delta, y = initial.clone(), y0
        errors = [loss(y)]
        for k in range(BUDGET):
            delta = (delta + 0.8 * (target - y)) * mask
            y = rollout(model, diff, states, rng, delta, active)
            errors.append(loss(y))
        record('fixed_feedback', y, {'error_curve': errors, 'seconds': time.time() - start, 'extra_nfe': BUDGET * (diff.num_timesteps - ec.START)})
        start = time.time()
        delta, y = initial.clone(), y0
        history = [(delta.clone(), (target - y) * mask)]
        eta = 0.8
        accepted_errors, trials = [loss(y)], []
        for k in range(BUDGET):
            residual = (target - y) * mask
            proposal = delta + eta * residual
            used_aa = False
            if len(history) >= 2:
                recent = history[-3:]
                residuals = torch.stack([r.flatten() for _, r in recent], dim=1).double()
                gram = residuals.T @ residuals
                gram += torch.eye(len(recent), device='cuda', dtype=torch.float64) * (0.001 * gram.diag().mean().clamp(min=1e-10))
                ones = torch.ones(len(recent), device='cuda', dtype=torch.float64)
                c = torch.linalg.solve(gram, ones)
                if abs(float(c.sum())) > 1e-10:
                    c = c / c.sum()
                    if float(c.abs().sum()) <= 4.:
                        proposal = sum(float(w) * (d + eta * r) for w, (d, r) in zip(c, recent))
                        used_aa = True
            proposal = proposal * mask
            candidate = rollout(model, diff, states, rng, proposal, active)
            err = loss(candidate)
            accept = bool(torch.isfinite(candidate).all()) and err < loss(y)
            trials.append({'error': err, 'accepted': accept, 'eta': eta, 'anderson': used_aa})
            if accept:
                delta, y = proposal, candidate
                history.append((delta.clone(), (target - y) * mask))
                history = history[-3:]
            else:
                eta = max(0.05, eta * 0.5)
                history = [(delta.clone(), (target - y) * mask)]
            accepted_errors.append(loss(y))
        record('safeguarded_feedback', y, {'error_curve': accepted_errors, 'trials': trials, 'seconds': time.time() - start, 'extra_nfe': BUDGET * (diff.num_timesteps - ec.START)})
        keys = ['base', 'hard_direct', 'fixed_feedback', 'safeguarded_feedback', 'target_composite']
        frames = []
        for f in range(16):
            strip = torch.cat([(videos[k][f] + 1) / 2 for k in keys], dim=2)
            arr = (strip.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
            frame = Image.new('RGB', (1280, 284), 'white')
            frame.paste(Image.fromarray(arr), (0, 28))
            draw = ImageDraw.Draw(frame)
            for j, key in enumerate(keys):
                draw.text((j * 256 + 6, 6), key, fill='black')
            frames.append(frame)
        frames[0].save(OUT / f'seed{seed}.gif', save_all=True, append_images=frames[1:], duration=125, loop=0)
        frames[8].save(OUT / f'seed{seed}.png')
        torch.save({'videos': videos}, OUT / f'seed{seed}.pt')
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
