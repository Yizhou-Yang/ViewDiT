#!/usr/bin/env python3
"""Edit-vs-Preserve CONFLICT probe (method-core audit, Round 2).

The prior audit (P1/P2/P3) split by frame: prefix=zA (preserve), future=zB (edit),
so the calibration never *conflicts* with the edit on the SAME frames. This probe
makes the two objectives collide on the SAME prefix frames and asks whether the
low-rank boundary state finds a solution INSIDE the subspace that simultaneously
(1) keeps semantic A (drives zB[:p] back toward zA[:p]) and
(2) lets the edit direction (zA-zB) survive (does not fully erase it).

Concretely, on the same prefix frames zB[:p], for each rank k we apply the rank-k
SVD correction dr toward zA[:p] and measure:
  keep_err      = ||z[:p] - zA[:p]||_rms                (preserve quality)
  edit_signal   = decomposition of the observed edit (zA[:p]-zB[:p]) into the part
                  aligned with the applied correction vs the residual the boundary
                  leaves free. residual_frac>0 @ low k => the boundary is NOT simply
                  pasting the prefix (selective trade-off signature); aligned_frac
                  tracks how much of the edit direction gets "kept" by the boundary.

References (measured on the SAME prefix frames):
  none         : zB[:p] untouched   -> keep_err=max, residual_frac~1 (edit free)
  full_oracle  : zB[:p]=zA[:p]      -> keep_err=0,   but residual_frac~0 (edit erased
                 = pure cache baseline)

Interpretation guard: if boundary_rank{k} == full_oracle for all k, it is just
compressed-prefix-cache (erases the observed edit on protected frames). A regime
where keep_err~0 AND residual_frac>0 at modest k is the selective trade-off we are
looking for. We do NOT over-claim: single-DiT mechanism check, not a general result.

Runs on V100 (/root/viewdit) via /root/miniconda3/bin/python.
"""
from __future__ import annotations

import json
import importlib.util
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(os.environ.get("VIEWDIT_ROOT", "/root/viewdit"))
LATTE = Path(os.environ.get("LATTE_ROOT", ROOT / "third_party/Latte"))
CKPT_DIR = Path(os.environ.get("LATTE_CKPT", ROOT / "weights/Latte"))
OUT = Path(os.environ.get("VIEWDIT_VIDEO_OUT", ROOT / "results/video/latte_conflict_probe"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(LATTE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_latte = _load("latte_core", LATTE / "models" / "latte.py")
Latte_models = _latte.Latte_models
from diffusion import create_diffusion  # noqa: E402


def find_model(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "ema" in ckpt:
        return ckpt["ema"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def rankk_dr(dev, k):
    """rank-k correction driving zB[:p] toward zA[:p]: top-k SVD of dev=(zA-zB)."""
    b, P, C, H, W = dev.shape
    flat = dev.reshape(b * P * C, H * W).t()
    U, Sv, Vt = torch.linalg.svd(flat.float(), full_matrices=False)
    k = min(k, U.shape[1])
    Uk = U[:, :k]
    coeff = Uk.t() @ flat
    dr = (Uk @ coeff).t().reshape(dev.shape)
    return dr


def main():
    model_name = os.environ.get("LATTE_MODEL", "Latte-XL/2")
    ckpt_name = os.environ.get("LATTE_CKPT_FILE", "ucf101.pt")
    steps = int(os.environ.get("LATTE_STEPS", "12"))
    frames = int(os.environ.get("LATTE_FRAMES", "16"))
    prefix = int(os.environ.get("LATTE_PREFIX", "6"))
    cfg = float(os.environ.get("LATTE_CFG", "7.0"))
    num_classes = int(os.environ.get("LATTE_CLASSES", "101"))
    class_a = int(os.environ.get("LATTE_CLASS_A", "14"))
    class_b = int(os.environ.get("LATTE_CLASS_B", "95"))
    seed = int(os.environ.get("LATTE_SEED", "999"))
    ranks = [int(x) for x in os.environ.get("LATTE_RANKS", "2,4,8,16,32,64").split(",") if x.strip()]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)

    args = Namespace(model=model_name, latent_size=32, num_classes=num_classes,
                     num_frames=frames, learn_sigma=True, extras=2)
    model = Latte_models[model_name](
        input_size=args.latent_size, num_classes=args.num_classes,
        num_frames=args.num_frames, learn_sigma=args.learn_sigma, extras=args.extras,
    ).to(device)
    model.load_state_dict(find_model(str(CKPT_DIR / ckpt_name)))
    model.eval()
    diffusion = create_diffusion(str(steps))

    g = torch.Generator(device=device).manual_seed(seed)
    z0 = torch.randn(1, frames, 4, 32, 32, generator=g, device=device, dtype=torch.float32)
    using_cfg = cfg > 1.0

    def sample(class_id, z, tag):
        if using_cfg:
            zb = torch.cat([z, z], dim=0)
            y = torch.tensor([class_id], device=device)
            y_null = torch.tensor([num_classes], device=device)
            y = torch.cat([y, y_null], dim=0)
            kwargs = dict(y=y, cfg_scale=cfg, use_fp16=True)
            fn = model.forward_with_cfg
        else:
            zb = z
            y = torch.tensor([class_id], device=device)
            kwargs = dict(y=y, use_fp16=True)
            fn = model
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = diffusion.ddim_sample_loop(
                fn, zb.shape, zb, clip_denoised=False,
                model_kwargs=kwargs, progress=False, device=zb.device,
            )
        x = out[:1].float()
        print(f"[{tag}] class={class_id} cfg={cfg} shape={tuple(x.shape)}", flush=True)
        return x

    zA = sample(class_a, z0, "base_classA")
    zB = sample(class_b, z0, "edit_classB")
    p = min(prefix, frames - 1)

    dev = zA[:, :p] - zB[:, :p]                  # correction target (same prefix frames)
    b, P, C, H, W = dev.shape
    dev_flat = dev.reshape(b * P * C, H * W).t()  # (H*W, bPC)
    observed_edit_norm = dev_flat.norm().item()

    def keep_err(z):
        return (z[:, :p] - zA[:, :p]).square().mean().sqrt().item()

    def edit_signal_survival(z):
        applied = (z[:, :p] - zB[:, :p]).reshape(-1).double()
        ref = (zA[:, :p] - zB[:, :p]).reshape(-1).double()
        denom = ref.norm().item() + 1e-9
        al = (ref @ applied) / (applied.norm().item() ** 2 + 1e-9)
        aligned = al * applied
        residual = ref - aligned
        return {
            "residual_frac": residual.norm().item() / denom,
            "aligned_frac": aligned.norm().item() / denom,
        }

    rows = []
    rows.append({"method": "none", "prefix_keep_err": keep_err(zB),
                 "edit_signal": edit_signal_survival(zB)})
    z_full = zB.clone()
    z_full[:, :p] = zA[:, :p].clone()
    rows.append({"method": "full_oracle", "prefix_keep_err": keep_err(z_full),
                 "edit_signal": edit_signal_survival(z_full)})

    maxc = min(dev_flat.shape[0], dev_flat.shape[1])
    for kk in ranks:
        k = min(kk, maxc)
        if k < 1:
            continue
        dr = rankk_dr(dev, k)
        zb = zB.clone()
        zb[:, :p] = zB[:, :p].clone() + dr
        rows.append({
            "method": f"boundary_rank{k}",
            "prefix_keep_err": keep_err(zb),
            "edit_signal": edit_signal_survival(zb),
        })
        print(f"ROW boundary_rank{k} keep_err={keep_err(zb):.6g} "
              f"resid={edit_signal_survival(zb)['residual_frac']:.4f} "
              f"aligned={edit_signal_survival(zb)['aligned_frac']:.4f}",
              flush=True)

    report = {
        "complete": True,
        "protocol": {
            "model": model_name, "ckpt": ckpt_name, "steps": steps,
            "frames": frames, "prefix": p, "ranks": ranks,
            "class_a": class_a, "class_b": class_b, "seed": seed, "cfg": cfg,
            "note": "Round-2 method-core probe: edit-vs-preserve conflict on the SAME "
                    "prefix frames. Checks whether low-rank boundary separates "
                    "'keep A' from 'release B' inside the subspace, instead of either "
                    "ignoring the conflict (none) or blindly pasting zA prefix "
                    "(full_oracle = pure cache).",
        },
        "observed_edit_norm_prefix": observed_edit_norm,
        "rows": rows,
        "interpretation": (
            "If boundary_rank{k>=some} hits keep_err~0 AND residual_frac>0 at modest k, "
            "the subspace does a selective trade-off rather than pure cache. "
            "If boundary_rank ~ full_oracle across all k, it is compressed-prefix-cache "
            "(erases the observed edit on protected frames)."
        ),
    }
    (OUT / "stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("LATTE_CONFLICT_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()