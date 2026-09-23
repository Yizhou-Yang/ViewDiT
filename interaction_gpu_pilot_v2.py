#!/usr/bin/env python3
"""Controlled mechanism pilot, NOT a foundation-model or real-video benchmark.
Requires torch 2.6 and numpy; optional Pillow for contact sheets.
Render: analytic 3D sphere/plane ray intersections, 4 directional lights,
Lambertian direct lighting, hard per-light visibility, no indirect illumination.
Train: source RGB + visible object masks + coordinates -> 3 edited videos.
Neither hidden geometry nor target differences are model inputs.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@torch.no_grad()
def scene(seed, h=48, w=64, frames=5):
    g = torch.Generator(device='cuda').manual_seed(seed)
    def rand(*shape):
        return torch.rand(shape, device='cuda', generator=g)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device='cuda'), torch.linspace(-1.4, 1.4, w, device='cuda'), indexing='ij')
    cam = torch.tensor([0., -6., 5.], device='cuda')
    forward = F.normalize(-cam, dim=0)
    right = torch.tensor([1., 0., 0.], device='cuda')
    up = F.normalize(torch.linalg.cross(right, forward), dim=0)
    origins = cam + 1.6 * xx[..., None] * right - 1.6 * yy[..., None] * up
    origins = origins.unsqueeze(0).expand(frames, -1, -1, -1)
    rays = forward.expand_as(origins)
    radii = .28 + .20 * rand(2)
    centers = torch.zeros(frames, 2, 3, device='cuda')
    centers[:, :, 2] = radii
    base = torch.stack([torch.tensor([-.50, .0], device='cuda'), torch.tensor([.50, .0], device='cuda')])
    base += (rand(2, 2) - .5) * .5
    velocity = (rand(2, 2) - .5) * .25
    centers[:, :, :2] = base[None] + torch.linspace(-1, 1, frames, device='cuda')[:, None, None] * velocity[None]
    colors = .15 + .70 * rand(2, 3)
    floor_color = .35 + .4 * rand(3)
    lights = F.normalize(torch.tensor([[-.65, -.45, 1.2], [-.50, -.45, 1.2], [-.65, -.30, 1.2], [-.50, -.30, 1.2]], device='cuda'), dim=-1)

    def intersect(o, d, center, radius):
        delta = o - center[:, None, None, :]
        b = (delta * d).sum(-1)
        disc = b.square() - delta.square().sum(-1) + radius.square()
        root = -b - disc.clamp_min(0).sqrt()
        return torch.where((disc > 0) & (root > 1e-4), root, torch.full_like(root, 1e6))

    def render(remove_a, remove_b):
        plane_t = -origins[..., 2] / rays[..., 2]
        distances = [plane_t]
        for i, removed in enumerate((remove_a, remove_b)):
            distances.append(torch.full_like(plane_t, 1e6) if removed else intersect(origins, rays, centers[:, i], radii[i]))
        depth, ids = torch.stack(distances).min(0)
        points = origins + depth[..., None] * rays
        normals = torch.zeros_like(points)
        normals[..., 2] = 1
        pattern = .92 + .08 * torch.sin(points[..., 0] * 3) * torch.cos(points[..., 1] * 3)
        albedo = pattern[..., None] * floor_color
        for i in range(2):
            m = (ids == i + 1)[..., None]
            normals = torch.where(m, F.normalize(points - centers[:, i, None, None, :], dim=-1), normals)
            albedo = torch.where(m, colors[i], albedo)
        intensity = torch.full_like(depth, .20)
        for light in lights:
            shadow_origin = points + normals * 1e-3
            blocked = torch.zeros_like(depth, dtype=torch.bool)
            for i, removed in enumerate((remove_a, remove_b)):
                if not removed:
                    blocked |= intersect(shadow_origin, light.expand_as(points), centers[:, i], radii[i]) < 1e5
            intensity += .20 * (normals * light).sum(-1).clamp_min(0) * (~blocked)
        return (albedo * intensity[..., None]).permute(3, 0, 1, 2).contiguous(), ids

    source, ids = render(0, 0)
    targets = torch.stack([render(1, 0)[0], render(0, 1)[0], render(1, 1)[0]])
    masks = torch.stack([ids == 1, ids == 2]).float()
    coords = torch.stack([xx.expand(frames, -1, -1), yy.expand(frames, -1, -1), torch.linspace(-1, 1, frames, device='cuda')[:, None, None].expand(-1, h, w)])
    inp = torch.cat([source, masks, coords])
    return inp, targets


class Editor(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.enc = nn.Sequential(nn.Conv3d(8, 24, 3, padding=1), nn.SiLU(), nn.Conv3d(24, 24, 3, padding=1), nn.SiLU())
        self.low = nn.Sequential(nn.Conv3d(24, 48, 3, padding=1), nn.SiLU(), nn.Conv3d(48, 48, 3, padding=1), nn.SiLU())
        self.dec = nn.Sequential(nn.Conv3d(72, 24, 3, padding=1), nn.SiLU())
        self.head = nn.Conv3d(24, 9, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e = self.enc(x)
        z = self.low(F.avg_pool3d(e, (1, 2, 2)))
        z = self.dec(torch.cat([e, F.interpolate(z, size=e.shape[2:], mode='trilinear', align_corners=False)], 1))
        r = self.head(z).reshape(x.shape[0], 3, 3, *x.shape[2:])
        if self.kind == 'interaction':
            r = torch.stack([r[:, 0], r[:, 1], r[:, 0] + r[:, 1] + r[:, 2]], 1)
        return x[:, None, :3] + r


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    rows = []
    for i in range(len(x)):
        pred = model(x[i:i+1])[0]
        err = (pred - y[i]).square().mean(1)
        change = (y[i] - x[i, None, :3]).abs().amax(1) > .005
        visible_floor = (x[i, 3:5].sum(0) == 0)[None]
        effects = change & visible_floor
        keep = ~change
        target_interaction = y[i, 2] - y[i, 0] - y[i, 1] + x[i, :3]
        pred_interaction = pred[2] - pred[0] - pred[1] + x[i, :3]
        def mean_region(mask):
            return float(err[mask].mean()) if bool(mask.any()) else None
        rows.append({'mse': float(err.mean()), 'changed_mse': mean_region(change), 'effects_mse': mean_region(effects), 'keep_mse': mean_region(keep), 'interaction_mse': float((pred_interaction - target_interaction).square().mean()), 'temporal_delta_mse': float(((pred[:, :, 1:] - pred[:, :, :-1]) - (y[i, :, :, 1:] - y[i, :, :, :-1])).square().mean()), 'effects_pixels': int(effects.sum())})
    return rows


def means(rows):
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--train', type=int, default=200)
    p.add_argument('--val', type=int, default=40)
    p.add_argument('--test', type=int, default=80)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'stats.json').exists():
        raise RuntimeError('Refuse to overwrite experiment; choose new out directory')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stats = {'complete': False, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'protocol': {'steps': args.steps, 'train_scenes': args.train, 'validation_scenes': args.val, 'test_scenes': args.test, 'frames': 5, 'height': 48, 'width': 64, 'batch': 4, 'seeds': [11, 29, 47], 'learning_rates': [0.001, 0.0003], 'selection': 'lowest validation changed-region MSE, across LR and checkpoints, separately per method and seed', 'objective': 'same full MSE + normalized changed/effects/keep MSE + 0.1 interaction MSE; train-only GT masks', 'limits': 'small Conv3D, analytic direct-light spheres, no foundation model, no real videos, no unseen operation combinations'}, 'runs': []}
    def save():
        tmp = args.out / 'stats.tmp'
        tmp.write_text(json.dumps(stats, indent=2))
        tmp.replace(args.out / 'stats.json')
    save()
    splits = {}
    for name, count, offset in [('train', args.train, 1000), ('val', args.val, 100000), ('test', args.test, 300000)]:
        data = [scene(offset + i) for i in range(count)]
        splits[name] = (torch.stack([v[0] for v in data]), torch.stack([v[1] for v in data]))
        del data
        print('rendered', name, count, flush=True)
    xtrain, ytrain = splits['train']
    xtest, ytest = splits['test']
    torch.save({'x': xtest[:4].cpu(), 'y': ytest[:4].cpu()}, args.out / 'test_examples.pt')
    stats['repeat_render_max_error'] = float((scene(1000)[0] - xtrain[0]).abs().max())
    inter = ytest[:, 2] - ytest[:, 0] - ytest[:, 1] + xtest[:, :3]
    stats['interaction_signal_mse'] = float(inter.square().mean())
    stats['interaction_present_scenes'] = int((inter.abs().flatten(1).amax(1) > .005).sum())
    class Identity(nn.Module):
        def forward(self, z):
            return z[:, None, :3].expand(-1, 3, -1, -1, -1, -1)
    stats['identity_baseline'] = means(evaluate(Identity(), xtest, ytest))
    torch.manual_seed(123)
    direct = Editor('direct').cuda()
    with torch.no_grad():
        direct.head.weight.normal_(0, .02)
        direct.head.bias.normal_(0, .02)
    structured = copy.deepcopy(direct)
    structured.kind = 'interaction'
    with torch.no_grad():
        for v in [structured.head.weight, structured.head.bias]:
            v[6:9].sub_(v[0:3] + v[3:6])
    stats['mapped_function_max_error'] = float((direct(xtest[:2]) - structured(xtest[:2])).abs().max().detach())
    stats['parameters_each'] = sum(p.numel() for p in direct.parameters())
    assert stats['mapped_function_max_error'] < 1e-5
    del direct, structured
    save()
    selected = {}
    for seed in [11, 29, 47]:
        for kind in ['direct', 'interaction']:
            best_score = float('inf')
            best_state = None
            best_config = None
            for lr in [0.001, 0.0003]:
                torch.manual_seed(seed)
                model = Editor(kind).cuda()
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                index_rng = torch.Generator(device='cuda').manual_seed(seed + 500)
                run = {'seed': seed, 'kind': kind, 'lr': lr, 'checks': []}
                torch.cuda.synchronize()
                start = time.perf_counter()
                for step in range(args.steps):
                    model.train()
                    idx = torch.randint(len(xtrain), (4,), device='cuda', generator=index_rng)
                    opt.zero_grad(set_to_none=True)
                    pred = model(xtrain[idx])
                    target = ytrain[idx]
                    error = (pred - target).square().mean(2)
                    changed = (target - xtrain[idx, None, :3]).abs().amax(2) > .005
                    effects = changed & (xtrain[idx, 3:5].sum(1) == 0)[:, None]
                    keep = ~changed
                    loss = error.mean() + error[changed].mean() + error[effects].mean() + error[keep].mean()
                    pi = pred[:, 2] - pred[:, 0] - pred[:, 1] + xtrain[idx, :3]
                    ti = target[:, 2] - target[:, 0] - target[:, 1] + xtrain[idx, :3]
                    loss = loss + 0.1 * F.mse_loss(pi, ti)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite loss')
                    loss.backward()
                    opt.step()
                    if (step + 1) % 200 == 0 or step + 1 == args.steps:
                        val = means(evaluate(model, *splits['val']))
                        run['checks'].append({'step': step + 1, 'train_batch_loss': float(loss.detach()), 'validation': val})
                        if val['changed_mse'] < best_score:
                            best_score = val['changed_mse']
                            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                            best_config = {'lr': lr, 'step': step + 1, 'validation_changed_mse': best_score}
                torch.cuda.synchronize()
                run['seconds_including_validation'] = time.perf_counter() - start
                stats['runs'].append(run)
                save()
                print(json.dumps({'finished': kind, 'seed': seed, 'lr': lr, 'seconds': run['seconds_including_validation'], 'validation': run['checks'][-1]['validation']}), flush=True)
                del model, opt, pred, loss
            model = Editor(kind).cuda()
            model.load_state_dict(best_state)
            rows = evaluate(model, *splits['test'])
            selected[f'{seed}_{kind}'] = {'configuration': best_config, 'mean': means(rows), 'per_scene': rows}
            torch.save({k: v.cpu() for k, v in best_state.items()}, args.out / f'{seed}_{kind}.pt')
            torch.save(model(xtest[:4]).detach().cpu(), args.out / f'{seed}_{kind}_predictions.pt')
            del model, best_state
    stats['selected'] = selected
    stats['bootstrap'] = {}
    rng = np.random.default_rng(20260922)
    for metric in ['mse', 'changed_mse', 'effects_mse', 'keep_mse', 'interaction_mse', 'temporal_delta_mse']:
        a = np.array([[selected[f'{s}_direct']['per_scene'][i][metric] for i in range(args.test)] for s in [11, 29, 47]], dtype=float)
        b = np.array([[selected[f'{s}_interaction']['per_scene'][i][metric] for i in range(args.test)] for s in [11, 29, 47]], dtype=float)
        valid = np.isfinite(a).all(0) & np.isfinite(b).all(0)
        a, b = a[:, valid], b[:, valid]
        boot = []
        for _ in range(2000):
            si = rng.integers(0, 3, 3)
            ii = rng.integers(0, a.shape[1], a.shape[1])
            aa, bb = a[si][:, ii].mean(), b[si][:, ii].mean()
            boot.append((aa - bb) / max(aa, 1e-12))
        stats['bootstrap'][metric] = {'direct': float(a.mean()), 'interaction': float(b.mean()), 'relative_reduction': float((a.mean() - b.mean()) / max(a.mean(), 1e-12)), 'paired_seed_scene_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(), 'independent_test_scenes': int(valid.sum())}
    stats['complete'] = True
    save()
    print(json.dumps(stats['bootstrap'], indent=2), flush=True)


if __name__ == '__main__':
    main()
