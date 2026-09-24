#!/usr/bin/env python3
"""Seven-direction same-budget screen on official Latte-XL/2 (FFS, DDIM-50).

Budget: 15 full-forward equivalents (FFE) per video; FFE = computed transformer
blocks / 28 (embed + final layer counted as free, they are <1% of FLOPs).
Reference: Full DDIM-50 from the same noise. Metrics: final-latent relL2 and
decoded LPIPS(alex, mean over 16 frames). Paired per-seed statistics.

Directions (none of these were measured in AUDIT_20260920 / CANDIDATES):
 D1 SCHED   non-uniform hit placement for distance-scaled Taylor-1
 D2 RETRO   zero-NFE retrospective interpolation correction of the state at hits
 D3 BLOCK   feature-level residual reuse (embed/head recomputed; m shallow pairs real)
 D4 HYBRID  as D3 but the cached deep residual is Taylor-extrapolated
 D5 STASYM  spatial-only steps reusing temporal-block residuals (Latte-specific)
            with a depth-half comparator at identical cost and positions
 D6 LEARN   per-step Taylor coefficients fitted on 8 separate train seeds
 D7 SOLVER  DPM-Solver++(2M), 15 NFE on the same grid
Baselines: reuse (zero-order), taylor1 (repo form), taylor1s (distance-scaled).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("LATTE_STEPS", "50")
os.environ.setdefault("LATTE_SAMPLER", "ddim")
os.environ.setdefault("LATTE_FP16", "1")
os.environ.setdefault("LATTE_K", "15")

from einops import rearrange, repeat  # noqa: E402
from run_xl_gate import FRAMES, _noise, load_model, rel_l2  # noqa: E402

OUT = Path(os.environ.get("SEVEN_OUT", "/root/viewdit/results/video/seven_dir_20260924"))
N_TEST = int(os.environ.get("SEVEN_N", "16"))
N_TRAIN = int(os.environ.get("SEVEN_TRAIN", "8"))
ONLY = os.environ.get("SEVEN_ONLY", "")
T = 50


class Net:
    def __init__(self, model, diffusion, device):
        self.m = model
        self.L = len(model.blocks)
        assert self.L % 2 == 0
        self.P = self.L // 2
        tm = getattr(diffusion, "timestep_map", None)
        self.tmap = list(tm) if tm is not None else list(range(T))
        self.device = device

    def t(self, idx):
        return torch.full((1,), int(self.tmap[idx]), device=self.device, dtype=torch.long)

    def embed(self, x, idx):
        m = self.m
        b = x.shape[0]
        h = rearrange(x.half(), "b f c h w -> (b f) c h w")
        h = m.x_embedder(h) + m.pos_embed
        te = m.t_embedder(self.t(idx), use_fp16=True)
        c_sp = repeat(te, "n d -> (n c) d", c=m.temp_embed.shape[1])
        c_tp = repeat(te, "n d -> (n c) d", c=m.pos_embed.shape[1])
        return h, c_sp, c_tp, b

    def pair(self, i, h, c_sp, c_tp, b, tcache=None):
        m = self.m
        sb, tb = m.blocks[2 * i], m.blocks[2 * i + 1]
        h = sb(h, c_sp)
        h = rearrange(h, "(b f) t d -> (b t) f d", b=b)
        if i == 0:
            h = h + m.temp_embed
        if tcache is None:
            out = tb(h, c_tp)
            res = out - h
            h = out
        else:
            res = tcache[i]
            h = h + res
        h = rearrange(h, "(b t) f d -> (b f) t d", b=b)
        return h, res

    def head(self, h, c_sp, b):
        m = self.m
        h = m.final_layer(h, c_sp)
        h = m.unpatchify(h)
        return rearrange(h, "(b f) c h w -> b f c h w", b=b)[:, :, :4].float()

    def full(self, x, idx, keep=()):
        h, c_sp, c_tp, b = self.embed(x, idx)
        snaps, tres = {}, []
        if 0 in keep:
            snaps[0] = h
        for i in range(self.P):
            h, r = self.pair(i, h, c_sp, c_tp, b)
            tres.append(r)
            if (i + 1) in keep:
                snaps[i + 1] = h
        deltas = {mp: h - snaps[mp] for mp in keep}
        return self.head(h, c_sp, b), deltas, tres

    def partial(self, x, idx, mp, delta):
        h, c_sp, c_tp, b = self.embed(x, idx)
        for i in range(mp):
            h, _ = self.pair(i, h, c_sp, c_tp, b)
        return self.head(h + delta, c_sp, b)

    def spatial_only(self, x, idx, tcache):
        h, c_sp, c_tp, b = self.embed(x, idx)
        for i in range(self.P):
            h, _ = self.pair(i, h, c_sp, c_tp, b, tcache=tcache)
        return self.head(h, c_sp, b)


def ddim_ab(diffusion):
    ab = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    prev = np.asarray(diffusion.alphas_cumprod_prev, dtype=np.float64)
    a = np.sqrt(prev / np.clip(ab, 1e-12, None))
    b = np.sqrt(np.clip(1.0 - prev, 0.0, None)) - a * np.sqrt(np.clip(1.0 - ab, 0.0, None))
    return a, b


def uniform_hits(k):
    return sorted(set(int(v) for v in np.linspace(0, T - 1, k)) | {0, T - 1})


def power_hits(k, p):
    u = np.linspace(0, 1, k)
    raw = [int(round((T - 1) * (q ** p))) for q in u]
    out = []
    for v in raw:
        while v in out:
            v += 1
        out.append(min(v, T - 1))
    out = sorted(set(out) | {0, T - 1})
    cand = [i for i in range(T) if i not in out]
    while len(out) < k:
        out = sorted(out + [cand.pop(len(cand) // 2)])
    while len(out) > k:
        out.pop(len(out) // 2)
    return out


def spread_extra(base, n):
    rest = [i for i in range(T) if i not in base]
    pick = [rest[int(round(v))] for v in np.linspace(0, len(rest) - 1, n)]
    return sorted(set(pick))


@torch.no_grad()
def run_plan(net, z, a, b, plan, fore="taylor1s", retro=False, keep=(), dfore=False, coef=None, record=False):
    """plan[i] in {'F', ('P', mp), 'S', '-'}; returns final latent, FFE cost, optional eps record."""
    x = z.float().clone()
    evals = []          # (i, eps)
    pend = []           # (k, eps_hat) forecast steps since last eval
    alog = {}
    dhist = {mp: [] for mp in keep}  # (i, delta)
    tcache = None
    cost = 0.0
    rec = []
    for i in range(T):
        idx = T - 1 - i
        act = plan[i]
        if act == "F":
            eps, deltas, tres = net.full(x, idx, keep=keep)
            tcache = tres
            for mp, d in deltas.items():
                dhist[mp] = (dhist[mp] + [(i, d)])[-2:]
            cost += 1.0
        elif isinstance(act, tuple) and act[0] == "P":
            mp = act[1]
            hs = dhist[mp]
            d = hs[-1][1]
            if dfore and len(hs) == 2:
                (j1, d1), (j, dj) = hs
                d = dj + ((i - j) / (j - j1)) * (dj - d1)
            eps = net.partial(x, idx, mp, d)
            cost += 2.0 * mp / net.L
        elif act == "S":
            eps = net.spatial_only(x, idx, tcache)
            cost += 0.5
        else:
            eps = None
        if eps is not None:
            if retro and pend and evals:
                j, ej = evals[-1]
                dx = torch.zeros_like(x)
                for k, eh in pend:
                    pk = 1.0
                    for mm in range(k + 1, i):
                        pk *= alog[mm]
                    et = ej + ((k - j) / (i - j)) * (eps - ej)
                    dx += pk * float(b[T - 1 - k]) * (et - eh)
                x = x + dx
            pend = []
            evals = (evals + [(i, eps)])[-2:]
        else:
            j, ej = evals[-1]
            if len(evals) < 2 or fore == "reuse":
                eps = ej
            else:
                j1, e1 = evals[-2]
                if fore == "taylor1":
                    s = 1.0
                elif fore == "learn":
                    s = coef.get(i, 1.0)
                else:
                    s = (i - j) / (j - j1)
                eps = ej + s * (ej - e1)
            pend.append((i, eps))
        if record:
            rec.append(eps.cpu())
        at, bt = float(a[idx]), float(b[idx])
        alog[i] = at
        x = at * x + bt * eps
    return x, cost, rec


@torch.no_grad()
def dpmpp2m(net, z, diffusion, hits):
    ab = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    seq = [T - 1 - i for i in hits]
    x = z.float().clone()
    x0p = hp = None
    for n, idx in enumerate(seq):
        eps, _, _ = net.full(x, idx)
        at, st = math.sqrt(ab[idx]), math.sqrt(1 - ab[idx])
        x0 = (x - st * eps) / at
        if n + 1 == len(seq):
            return x0, float(len(seq))
        idx2 = seq[n + 1]
        an, sn = math.sqrt(ab[idx2]), math.sqrt(1 - ab[idx2])
        h = math.log(an / sn) - math.log(at / st)
        if x0p is None:
            d = x0
        else:
            r = hp / h
            d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * x0p
        x = (sn / st) * x - an * (math.exp(-h) - 1) * d
        x0p, hp = x0, h
    return x, float(len(seq))


def variants():
    h15 = uniform_hits(15)
    full = ["F"] * T

    def hitplan(h):
        return ["F" if i in h else "-" for i in range(T)]

    v = {}
    v["B0_reuse"] = dict(plan=hitplan(h15), fore="reuse")
    v["B1_taylor1"] = dict(plan=hitplan(h15), fore="taylor1")
    v["B2_taylor1s"] = dict(plan=hitplan(h15), fore="taylor1s")
    v["D1_sched_p0.75"] = dict(plan=hitplan(power_hits(15, 0.75)), fore="taylor1s")
    v["D1_sched_p1.5"] = dict(plan=hitplan(power_hits(15, 1.5)), fore="taylor1s")
    v["D2_retro_taylor1s"] = dict(plan=hitplan(h15), fore="taylor1s", retro=True)
    v["D2_retro_reuse"] = dict(plan=hitplan(h15), fore="reuse", retro=True)
    p0 = ["F" if i in h15 else ("P", 0) for i in range(T)]
    v["D3_block_m0"] = dict(plan=p0, keep=(0,))
    h9 = uniform_hits(9)
    p2 = ["F" if i in h9 else ("P", 2) for i in range(T)]
    v["D3_block_m2"] = dict(plan=p2, keep=(2,))
    v["D4_hybrid_m0"] = dict(plan=p0, keep=(0,), dfore=True)
    v["D4_hybrid_m2"] = dict(plan=p2, keep=(2,), dfore=True)
    h8 = uniform_hits(8)
    ex = spread_extra(h8, 14)
    v["D5_stasym"] = dict(plan=["F" if i in h8 else ("S" if i in ex else "-") for i in range(T)], fore="taylor1s")
    v["D5_depthhalf"] = dict(plan=["F" if i in h8 else (("P", 7) if i in ex else "-") for i in range(T)], fore="taylor1s", keep=(7,))
    v["D6_learn"] = dict(plan=hitplan(h15), fore="learn")
    v["D7_dpmpp2m"] = dict(solver=h15)
    return full, v


def fit_coef(net, a, b, diffusion, device, hits):
    num, den = {}, {}
    full = ["F"] * T
    for s in range(N_TRAIN):
        z = _noise(device, 7000 + s)
        _, _, rec = run_plan(net, z, a, b, full, record=True)
        E = [r.double() for r in rec]
        for k in range(T):
            if k in hits:
                continue
            prior = [h for h in hits if h < k]
            if len(prior) < 2:
                continue
            j, j1 = prior[-1], prior[-2]
            d = E[j] - E[j1]
            num[k] = num.get(k, 0.0) + float(((E[k] - E[j]) * d).sum())
            den[k] = den.get(k, 0.0) + float((d * d).sum())
        print("fit seed", s, flush=True)
    return {k: num[k] / max(den[k], 1e-12) for k in num}


def lpips_video(fn, vae, x, ref_imgs):
    imgs = decode(vae, x)
    with torch.no_grad():
        return float(fn(imgs, ref_imgs).mean()), imgs


def decode(vae, x):
    lat = x[0].half() / 0.18215
    out = []
    with torch.no_grad():
        for i in range(0, lat.shape[0], 4):
            out.append(vae.decode(lat[i:i + 4]).sample.float().clamp(-1, 1))
    return torch.cat(out, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    model, diffusion = load_model(device)
    net = Net(model, diffusion, device)
    a, b = ddim_ab(diffusion)
    assert len(a) == T, len(a)

    # equivalence check: custom full forward vs the official wrapped model
    wrapped = diffusion._wrap_model(model)
    z0 = _noise(device, 1)
    ref = wrapped(z0, torch.full((1,), 30, device=device, dtype=torch.long), y=None, use_fp16=True)[:, :, :4].float()
    mine, _, _ = net.full(z0.float(), 30)
    eqerr = rel_l2(mine.cpu().numpy(), ref.cpu().numpy())
    print("equivalence relL2", eqerr, flush=True)
    assert eqerr < 1e-3

    full_plan, V = variants()
    if ONLY:
        V = {k: v for k, v in V.items() if k in ONLY.split(",")}
    coef = fit_coef(net, a, b, diffusion, device, uniform_hits(15)) if "D6_learn" in V else {}
    (OUT / "learn_coef.json").write_text(json.dumps(coef, indent=1))

    from diffusers.models import AutoencoderKL
    import lpips
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to(device).eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    rows = {k: {"relL2": [], "lpips": [], "cost": [], "sec": []} for k in V}
    meta = {"budget_ffe": 15, "steps": T, "frames": FRAMES, "test_seeds": list(range(9000, 9000 + N_TEST)),
            "train_seeds": list(range(7000, 7000 + N_TRAIN)), "equivalence_relL2": eqerr,
            "plans": {k: [p if isinstance(p, str) else f"P{p[1]}" for p in v.get("plan", [])] for k, v in V.items()}}
    for s in range(N_TEST):
        z = _noise(device, 9000 + s)
        t0 = time.time()
        xf, _, _ = run_plan(net, z, a, b, full_plan)
        tf = time.time() - t0
        ref_imgs = decode(vae, xf)
        for k, v in V.items():
            torch.cuda.synchronize()
            t0 = time.time()
            if "solver" in v:
                xk, cost = dpmpp2m(net, z, diffusion, v["solver"])
            else:
                xk, cost, _ = run_plan(net, z, a, b, v["plan"], fore=v.get("fore", "taylor1s"), retro=v.get("retro", False),
                                       keep=v.get("keep", ()), dfore=v.get("dfore", False), coef=coef)
            torch.cuda.synchronize()
            sec = time.time() - t0
            l, _ = lpips_video(lp, vae, xk, ref_imgs)
            r = rows[k]
            r["relL2"].append(rel_l2(xk.cpu().numpy(), xf.cpu().numpy()))
            r["lpips"].append(l)
            r["cost"].append(cost)
            r["sec"].append(sec)
            print(f"seed {s} {k} relL2={r['relL2'][-1]:.4f} lpips={l:.4f} ffe={cost:.2f} sec={sec:.1f} (full {tf:.1f}s)", flush=True)
        meta["full_sec_last"] = tf
        summarize(rows, meta, s + 1)


def summarize(rows, meta, n):
    base_keys = [k for k in rows if k.startswith("B")]
    res = {}
    for metric in ["relL2", "lpips"]:
        best = min(base_keys, key=lambda k: float(np.median(rows[k][metric])))
        bb = np.asarray(rows[best][metric])
        for k, r in rows.items():
            v = np.asarray(r[metric])
            d = (bb - v) / np.clip(bb, 1e-12, None)
            sem = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 1.0
            e = res.setdefault(k, {"cost": float(np.mean(r["cost"])), "sec": float(np.mean(r["sec"]))})
            e[metric] = {"median": float(np.median(v)), "best_baseline": best,
                         "drop_median": float(np.median(d)), "drop_mean": float(d.mean()), "sem": sem,
                         "win": float((v < bb).mean()),
                         "gate": bool(k not in base_keys and np.median(d) >= 0.25 and (v < bb).mean() >= 0.75 and d.mean() > 2 * sem)}
    out = {"meta": meta, "n_seeds": n, "summary": res, "per_seed": rows}
    (OUT / "stats.json").write_text(json.dumps(out, indent=1))
    lines = [f"# Seven-direction screen, Latte-XL/2 DDIM-50, budget 15 FFE, seeds={n}", "",
             "| variant | FFE | sec | relL2 med | drop vs best base | win | LPIPS med | drop | win | gate(relL2/LPIPS) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in res.items():
        r, l = e["relL2"], e["lpips"]
        lines.append(f"| {k} | {e['cost']:.2f} | {e['sec']:.1f} | {r['median']:.4f} | {r['drop_median']:+.3f} | {r['win']:.2f} | "
                     f"{l['median']:.4f} | {l['drop_median']:+.3f} | {l['win']:.2f} | {r['gate']}/{l['gate']} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
