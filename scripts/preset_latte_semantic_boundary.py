#!/usr/bin/env python3
"""Second-DiT (Latte-XL/2 on UCF101, class-conditional) semantic prefix preservation.

PURPOSE (option A): make the semantic edit strong and realistic (CFG), then show a
compact boundary-state keeps the committed prefix while the future is rewritten by
a different action class. Latent layout: (B, F, C=4, H=32, W=32). Probe only; pixel
decode deferred.
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
OUT = Path(os.environ.get("VIEWDIT_VIDEO_OUT", ROOT / "results/video/latte_semantic_boundary"))
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
        print("Using Ema!", flush=True)
        return ckpt["ema"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        print("Using model!", flush=True)
        return ckpt["model"]
    print("Using ckpt as-is!", flush=True)
    return ckpt


def main():
    model_name = os.environ.get("LATTE_MODEL", "Latte-XL/2")
    ckpt_name = os.environ.get("LATTE_CKPT_FILE", "ucf101.pt")
    steps = int(os.environ.get("LATTE_STEPS", "12"))
    frames = int(os.environ.get("LATTE_FRAMES", "16"))
    prefix = int(os.environ.get("LATTE_PREFIX", "8"))
    cfg = float(os.environ.get("LATTE_CFG", "7.0"))
    num_classes = int(os.environ.get("LATTE_CLASSES", "101"))
    class_a = int(os.environ.get("LATTE_CLASS_A", "14"))
    class_b = int(os.environ.get("LATTE_CLASS_B", "95"))
    seed = int(os.environ.get("LATTE_SEED", "999"))
    ranks = [int(x) for x in os.environ.get("LATTE_RANKS", "2,4,8,16,32,64").split(",") if x.strip()]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)

    args = Namespace(
        model=model_name,
        latent_size=32,
        num_classes=num_classes,
        num_frames=frames,
        learn_sigma=True,
        extras=2,
    )
    model = Latte_models[model_name](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        learn_sigma=args.learn_sigma,
        extras=args.extras,
    ).to(device)
    state = find_model(str(CKPT_DIR / ckpt_name))
    model.load_state_dict(state)
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
    edit_dist = (zA - zB).square().mean().sqrt().item()
    print(f"edit_distance_ab = {edit_dist:.6g}", flush=True)

    z_none = zB.clone()
    z_full = torch.cat([zA[:, :p].clone(), zB[:, p:].clone()], dim=1)

    dev = zA[:, :p] - zB[:, :p]
    b, P, C, Hh, Ww = dev.shape
    flat = dev.reshape(b * P * C, Hh * Ww).t()
    U, Sv, Vt = torch.linalg.svd(flat.float(), full_matrices=False)
    maxc = flat.shape[0]

    def prefix_err(z):
        return (z[:, :p] - zA[:, :p]).square().mean().sqrt().item()

    def future_err(z):
        return (z[:, p:] - zB[:, p:]).square().mean().sqrt().item()

    rows = [
        {"method": "none", "prefix_latent_rms": prefix_err(z_none),
         "future_latent_rms": future_err(z_none), "payload_bytes": None},
        {"method": "full_oracle", "prefix_latent_rms": prefix_err(z_full),
         "future_latent_rms": future_err(z_full),
         "payload_bytes": int(dev.numel() * dev.element_size())},
    ]
    for kk in ranks:
        k = min(kk, maxc)
        if k < 1:
            continue
        U_k = U[:, :k]
        coeff = U_k.t() @ flat
        dr = (U_k @ coeff).t().reshape(dev.shape)
        zb = zB.clone()
        zb[:, :p] = zB[:, :p].clone() + dr
        rows.append({
            "method": f"boundary_rank{k}",
            "prefix_latent_rms": prefix_err(zb),
            "future_latent_rms": future_err(zb),
            "payload_bytes": int(k * (Hh * Ww) * dev.element_size()) + k,
        })
        print(f"ROW boundary_rank{k} prefix_rms {round(prefix_err(zb),6)}", flush=True)

    report = {
        "complete": True,
        "edit_distance_ab": edit_dist,
        "cfg": cfg,
        "protocol": {
            "model": model_name, "ckpt": ckpt_name, "steps": steps,
            "frames": frames, "prefix": p, "ranks": ranks,
            "class_a": class_a, "class_b": class_b, "seed": seed,
            "note": "second DiT (Latte-XL/2, UCF101, extras=2) with CFG; regenerate future "
                    "under a very different action class = strong semantic-edit proxy; test "
                    "whether a compact rank-k boundary state preserves the committed prefix.",
            "decoder": "deferred pixel decode.",
        },
        "rows": rows,
    }
    (OUT / "stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("LATTE_SEMANTIC_BOUNDARY_OK", flush=True)


if __name__ == "__main__":
    main()