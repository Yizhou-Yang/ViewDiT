#!/usr/bin/env python3
"""Fixed-bridge convergence audit, not a new optimizer or perceptual benchmark."""
import argparse
import json
import time
from pathlib import Path
import torch
from wan_vae_official import WanVAE
from wan_state_probe import stream
from wan_tail_compensate import suffix
ROOT = Path('/root/viewdit/results/video/wan_real_pilot')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', type=Path, default=Path('/root/viewdit/results/video/wan_bridge_convergence'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    vae = WanVAE(vae_pth='/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth', dtype=torch.float32, device='cuda')
    report = {'complete': False, 'protocol': {'protected_start': 13, 'core_end': 7, 'bridge_frames': 6, 'frozen_latents': [0, 1, 2], 'adam_steps': 96, 'lbfgs_max_iter': 80, 'lbfgs_max_eval': 100, 'feasibility_relative_rms': .1, 'methods': ['adam_cold', 'adam_warm', 'lbfgs_warm'], 'warning': 'Exploratory convergence audit after 24-step results; not preregistered confirmation, equal time, or a novel optimizer. Warm-start encoding cost included, model loading and reference state setup excluded.'}, 'rows': []}
    def save(row=None):
        if row is not None:
            report['rows'].append(row)
            print('ROW', json.dumps(row), flush=True)
        temp = args.out / 'stats.tmp'
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.out / 'stats.json')
    paths = sorted(ROOT.glob('*.pt'))
    if args.limit:
        paths = paths[:args.limit]
    for path in paths:
        d = torch.load(path, map_location='cpu', weights_only=True)
        z = d['edited_z'].cuda()
        base = d['videos']['base'].cuda()
        edited = d['videos']['edit_latent_only'].cuda()
        with torch.no_grad():
            _, states = stream(vae, z)
            state = states[3]
            core = (edited[:, 5:7] - base[:, 5:7]).square().mean().sqrt()
            initial = (edited[:, 13:] - base[:, 13:]).square().mean().clamp(min=1e-12)
        for method in report['protocol']['methods']:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tic = time.time()
            with torch.no_grad():
                if method.endswith('warm'):
                    alpha = torch.ones(17, device='cuda')
                    alpha[7:13] = torch.linspace(1, 0, 8, device='cuda')[1:-1]
                    alpha[13:] = 0
                    target = base + alpha.view(1, 17, 1, 1) * (edited - base)
                    zr = vae.model.encode(target.unsqueeze(0), vae.scale)[0].float()
                    init = zr[:, 3:] - z[:, 3:]
                else:
                    init = torch.zeros_like(z[:, 3:])
            delta = init.detach().requires_grad_(True)
            curve = []
            counts = {'forwards': 0, 'backwards': 0}
            scale = z[:, 3:].square().mean().clamp(min=1e-8)
            if method.startswith('adam'):
                opt = torch.optim.Adam([delta], lr=.03)
            else:
                opt = torch.optim.LBFGS([delta], lr=1., max_iter=80, max_eval=100, history_size=10, tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(set_to_none=True)
                y = suffix(vae, z[:, 3:] + delta, state)
                mse = (y[:, 4:] - base[:, 13:]).square().mean()
                loss = mse / initial + .001 * delta.square().mean() / scale
                assert torch.isfinite(loss)
                loss.backward()
                counts['forwards'] += 1
                counts['backwards'] += 1
                curve.append(float(mse.detach().sqrt() / core))
                return loss
            if method.startswith('adam'):
                for _ in range(96):
                    closure()
                    opt.step()
            else:
                opt.step(closure)
            with torch.no_grad():
                zz = z.clone()
                zz[:, 3:] += delta
                final, _ = stream(vae, zz)
                counts['forwards'] += 1
                error = float((final[:, :7] - edited[:, :7]).abs().max())
                assert error == 0
                ratio = float((final[:, 13:] - base[:, 13:]).square().mean().sqrt() / core)
                torch.cuda.synchronize()
                save({'sample': path.stem, 'method': method, 'relative_protected_rms': ratio, 'passes_budget': ratio <= .1, 'initial_relative_protected_rms': float(initial.sqrt() / core), 'core_max_error': error, 'seconds': time.time() - tic, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), **counts, 'relative_latent_delta_rms': float(delta.square().mean().sqrt() / scale.sqrt()), 'curve': curve})
                torch.save({'latent': zz.cpu(), 'videos': {'base': base.cpu(), 'edit': edited.cpu(), 'bridge': final.cpu()}}, args.out / f'{path.stem}_{method}.pt')
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
