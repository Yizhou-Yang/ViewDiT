#!/usr/bin/env python3
"""FastVMT-style multi-component combos on Latte-XL/2 (FFS, DDIM-50), 15 FFE budget.

Component pool (7 most promising from all prior exploration):
  T1  Taylor-1 step forecast (repo form, strongest single baseline)
  T2  Taylor-2 step forecast (TaylorSeer order 2)
  R   retrospective interpolation correction at hits (weight w)
  SC  non-uniform hit schedule (power p)
  S   spatial-only anchor steps reusing temporal-block residuals (0.5 FFE)
  DR  deep-residual reuse with fresh shallow pairs (P, m)
  L   closed-loop learned per-step forecast coefficient
Each combo is at exactly (or <=) 15 FFE and compared against B1 (T1 alone).
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
import seven_dir_screen as S  # noqa: E402
from seven_dir_screen import Net, T, ddim_ab, decode, power_hits, uniform_hits, spread_extra  # noqa: E402
from run_xl_gate import _noise, load_model, rel_l2  # noqa: E402

OUT = Path(os.environ.get("COMBO_OUT", "/root/viewdit/results/video/seven_combo_20260924"))
N_TEST = int(os.environ.get("COMBO_N", "16"))
ONLY = os.environ.get("COMBO_ONLY", "")


@torch.no_grad()
def run2(net, z, a, b, plan, fore="taylor1", retro_w=0.0, keep=(), coef=None):
    x = z.float().clone()
    evals, pend, alog = [], [], {}
    dhist = {mp: None for mp in keep}
    tcache, cost = None, 0.0
    for i in range(T):
        idx = T - 1 - i
        act = plan[i]
        eps = None
        if act == "F":
            eps, deltas, tres = net.full(x, idx, keep=keep)
            tcache = tres
            for mp, d in deltas.items():
                dhist[mp] = d
            cost += 1.0
        elif act == "S":
            eps = net.spatial_only(x, idx, tcache)
            cost += 0.5
        elif isinstance(act, tuple) and act[0] == "P":
            eps = net.partial(x, idx, act[1], dhist[act[1]])
            cost += 2.0 * act[1] / net.L
        if eps is not None:
            if retro_w > 0 and pend and evals:
                j, ej = evals[-1]
                dx = torch.zeros_like(x)
                for k, eh in pend:
                    pk = 1.0
                    for mm in range(k + 1, i):
                        pk *= alog[mm]
                    et = ej + ((k - j) / (i - j)) * (eps - ej)
                    dx += pk * float(b[T - 1 - k]) * (et - eh)
                x = x + retro_w * dx
            pend = []
            evals = (evals + [(i, eps)])[-3:]
        else:
            j, ej = evals[-1]
            if len(evals) < 2:
                eps = ej
            else:
                j1, e1 = evals[-2]
                d1 = ej - e1
                if fore == "reuse":
                    eps = ej
                elif fore == "learn":
                    eps = ej + coef.get(str(i), coef.get(i, 1.0)) * d1
                elif fore == "taylor2" and len(evals) >= 3:
                    j2, e2 = evals[-3]
                    eps = ej + d1 + 0.5 * (d1 - (e1 - e2))
                else:
                    eps = ej + d1
            pend.append((i, eps))
        alog[i] = float(a[idx])
        x = float(a[idx]) * x + float(b[idx]) * eps
    return x, cost


def hitplan(h):
    return ["F" if i in h else "-" for i in range(T)]


def splan(nf, ns, sched=None):
    h = power_hits(nf, sched) if sched else uniform_hits(nf)
    ex = spread_extra(h, ns)
    return ["F" if i in h else ("S" if i in ex else "-") for i in range(T)]


def fit_closed_loop(net, a, b, device, plan, n=4):
    """greedy closed-loop per-step coefficient for T1 form: pick s in grid minimizing final relL2 on train seeds."""
    grid = [0.0, 0.5, 1.0, 1.5]
    coef = {}
    zs = [_noise(device, 7000 + s) for s in range(n)]
    refs = [run2(net, z, a, b, ["F"] * T)[0] for z in zs]
    gaps = [i for i in range(T) if plan[i] == "-"]
    for sweep in range(1):
        for k in gaps:
            best, bv = 1.0, 1e9
            for g in grid:
                c = dict(coef)
                c[k] = g
                v = np.mean([rel_l2(run2(net, z, a, b, plan, fore="learn", coef=c)[0].cpu().numpy(), r.cpu().numpy())
                             for z, r in zip(zs, refs)])
                if v < bv:
                    best, bv = g, v
            coef[k] = best
        print("closed-loop fit sweep", sweep, bv, flush=True)
    return coef


def variants():
    h15 = uniform_hits(15)
    v = {}
    v["B1_T1"] = dict(plan=hitplan(h15), fore="taylor1")
    v["B0_reuse_lp"] = dict(plan=hitplan(h15), fore="reuse")
    v["C01_T2"] = dict(plan=hitplan(h15), fore="taylor2")
    v["C02_T1+R1"] = dict(plan=hitplan(h15), fore="taylor1", retro_w=1.0)
    v["C03_T1+R.5"] = dict(plan=hitplan(h15), fore="taylor1", retro_w=0.5)
    v["C04_T1+SC1.25"] = dict(plan=hitplan(power_hits(15, 1.25)), fore="taylor1")
    v["C05_T1+SC1.5"] = dict(plan=hitplan(power_hits(15, 1.5)), fore="taylor1")
    v["C06_T1+SC1.25+R.5"] = dict(plan=hitplan(power_hits(15, 1.25)), fore="taylor1", retro_w=0.5)
    v["C07_T1+S(12F6S)"] = dict(plan=splan(12, 6), fore="taylor1")
    v["C08_T1+S(11F8S)"] = dict(plan=splan(11, 8), fore="taylor1")
    v["C09_T1+S+R.5"] = dict(plan=splan(12, 6), fore="taylor1", retro_w=0.5)
    v["C10_T2+S+R.5"] = dict(plan=splan(12, 6), fore="taylor2", retro_w=0.5)
    v["C11_T1+S+SC1.25+R.5"] = dict(plan=splan(12, 6, 1.25), fore="taylor1", retro_w=0.5)
    h13 = uniform_hits(13)
    ex = spread_extra(h13, 14)
    v["C12_T1+DR(13F14P2)"] = dict(plan=["F" if i in h13 else (("P", 1) if i in ex else "-") for i in range(T)],
                                   fore="taylor1", keep=(1,))
    v["C13_L_closed"] = dict(plan=hitplan(h15), fore="learn")
    v["C14_L_closed+R.5"] = dict(plan=hitplan(h15), fore="learn", retro_w=0.5)
    return v


def deep_variants():
    v = {}
    for K in (10, 15, 20):
        v[f"B1_T1@{K}"] = dict(plan=hitplan(uniform_hits(K)), fore="taylor1")
        for p in (1.0, 1.25):
            for w in (0.25, 0.5, 0.75):
                h = uniform_hits(K) if p == 1.0 else power_hits(K, p)
                v[f"C_p{p}_w{w}@{K}"] = dict(plan=hitplan(h), fore="taylor1", retro_w=w)
        v[f"C_T1+SC1.25@{K}"] = dict(plan=hitplan(power_hits(K, 1.25)), fore="taylor1")
    return v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    model, diffusion = load_model(device)
    net = Net(model, diffusion, device)
    a, b = ddim_ab(diffusion)
    V = variants()
    if os.environ.get("COMBO_VARIANTS") == "deep":
        V = deep_variants()
    SEED0 = int(os.environ.get("COMBO_SEED0", "9000"))
    if ONLY:
        V = {k: v for k, v in V.items() if k in ONLY.split(",") or k.startswith("B")}
    coef = {}
    if any(v["fore"] == "learn" for v in V.values()):
        coef = fit_closed_loop(net, a, b, device, hitplan(uniform_hits(15)))
        (OUT / "closed_coef.json").write_text(json.dumps(coef, indent=1))
    from diffusers.models import AutoencoderKL
    import lpips
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to(device).eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    rows = {k: {"relL2": [], "lpips": [], "cost": [], "sec": []} for k in V}
    meta = {"budget_ffe": "var", "test_seeds": list(range(SEED0, SEED0 + N_TEST)),
            "plans": {k: [p if isinstance(p, str) else f"P{p[1]}" for p in v["plan"]] for k, v in V.items()}}
    for s in range(N_TEST):
        z = _noise(device, SEED0 + s)
        xf, _ = run2(net, z, a, b, ["F"] * T)
        ref_imgs = decode(vae, xf)
        for k, v in V.items():
            torch.cuda.synchronize()
            t0 = time.time()
            xk, cost = run2(net, z, a, b, v["plan"], fore=v["fore"], retro_w=v.get("retro_w", 0.0),
                            keep=v.get("keep", ()), coef=coef)
            torch.cuda.synchronize()
            sec = time.time() - t0
            with torch.no_grad():
                l = float(lp(decode(vae, xk), ref_imgs).mean())
            r = rows[k]
            r["relL2"].append(rel_l2(xk.cpu().numpy(), xf.cpu().numpy()))
            r["lpips"].append(l)
            r["cost"].append(cost)
            r["sec"].append(sec)
            print(f"seed {s} {k} relL2={r['relL2'][-1]:.4f} lpips={l:.4f} ffe={cost:.2f}", flush=True)
        summarize(rows, meta, s + 1)


def summarize(rows, meta, n):
    res = {}
    for metric in ["relL2", "lpips"]:
        for k, r in rows.items():
            bk = ("B1_T1@" + k.split("@")[1]) if "@" in k else "B1_T1"
            bb = np.asarray(rows[bk][metric])
            v = np.asarray(r[metric])
            d = (bb - v) / np.clip(bb, 1e-12, None)
            sem = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 1.0
            e = res.setdefault(k, {"cost": float(np.mean(r["cost"]))})
            e[metric] = {"median": float(np.median(v)), "mean": float(v.mean()), "drop_median": float(np.median(d)),
                         "drop_mean": float(d.mean()), "sem": sem, "win": float((v < bb).mean()),
                         "gate": bool(k != bk and np.median(d) >= 0.25 and (v < bb).mean() >= 0.75 and d.mean() > 2 * sem)}
    (OUT / "stats.json").write_text(json.dumps({"meta": meta, "n_seeds": n, "summary": res, "per_seed": rows}, indent=1))
    lines = [f"# FastVMT-style combos vs B1_T1, Latte-XL/2 DDIM-50, 15 FFE, seeds={n}", "",
             "| variant | FFE | relL2 med | drop med | drop mean±sem | win | LPIPS med | drop med | drop mean±sem | win | gate r/l |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in res.items():
        r, l = e["relL2"], e["lpips"]
        lines.append(f"| {k} | {e['cost']:.2f} | {r['median']:.4f} | {r['drop_median']:+.3f} | {r['drop_mean']:+.3f}±{r['sem']:.3f} | {r['win']:.2f} | "
                     f"{l['median']:.4f} | {l['drop_median']:+.3f} | {l['drop_mean']:+.3f}±{l['sem']:.3f} | {l['win']:.2f} | {r['gate']}/{l['gate']} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
