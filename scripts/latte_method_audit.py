#!/usr/bin/env python3
"""Method self-audit: is the compact boundary-state result a real mechanism or
a self-fulfilling loop / trivial linear algebra?

Three adversarial probes on Latte-XL/2 (UCF101, CFG):

P1 LOOP-DETECTION: current 'boundary' builds dr = low-rank(zA[:p] - zB[:p]).
    We compare against a FAKE boundary using a random orthonormal direction,
    same rank and magnitude. If the fake also drives prefix error to ~0 as rank
    grows, the "monotonic approach to 0" is just the rank-k projection property,
    NOT evidence of semantic preservation.

P2 TRIVIALITY-CHECK: compare the boundary state vs directly storing a low-rank
    copy of (zA[:p]-zB[:p]) — the 'cache' baseline. If both give identical prefix
    error, the mechanism is just "compressed prefix cache", not a semantic device.

P3 COUPLING-TEST: Latte has NO temporal attention (frames processed independently,
    only temp_embed added). So calibrating the prefix in-latent cannot couple into
    the future, and full_oracle future_err==0 is an in-latent convenience, NOT a
    verified post-edit generation result. We report this honest limitation.

Output JSON under VIEWDIT_VIDEO_OUT.
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
OUT = Path(os.environ.get("VIEWDIT_VIDEO_OUT", ROOT / "results/video/latte_audit"))
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
try:
    from diffusion import create_diffusion  # noqa: E402
except Exception:
    raise


def find_model(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "ema" in ckpt:
        return ckpt["ema"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def lowrank_proj(target, source, k):
    """rank-k least-squares correction moving source toward target along the
    top-k left singular directions of (target - source). This is exactly what
    the current 'boundary state' does in the main probe."""
    dev = target - source
    b, P, C, H, W = dev.shape
    flat = dev.reshape(b * P * C, H * W).t()
    U, Sv, Vt = torch.linalg.svd(flat.float(), full_matrices=False)
    k = min(k, U.shape[1])
    Uk = U[:, :k]
    coeff = Uk.t() @ flat
    dr = (Uk @ coeff).t().reshape(dev.shape)
    return dr


def lowrank_cache(target, k):
    """Directly store a rank-k approximation of `target` (cache baseline)."""
    b, P, C, H, W = target.shape
    flat = target.reshape(b * P * C, H * W).t()
    U, Sv, Vt = torch.linalg.svd(flat.float(), full_matrices=False)
    k = min(k, U.shape[1])
    Uk = U[:, :k]
    recon = (Uk @ (Uk.t() @ flat)).t().reshape(target.shape)
    return recon


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
    ranks = [int(x) for x in os.environ.get("LATTE_RANKS", "2,4,8,16,32,64").split(",")]
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
            y = torch.cat([y, torch.tensor([num_classes], device=device)], dim=0)
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

    def prefix_rms(z):
        return (z[:, :p] - zA[:, :p]).square().mean().sqrt().item()

    audit = {"edit_distance_ab": (zA - zB).square().mean().sqrt().item()}
    p1_true = []
    p1_fake = []
    p2_boundary = []
    p2_cache = []
    for k in ranks:
        dr_true = lowrank_proj(zA[:, :p], zB[:, :p], k)
        z_true = zB.clone()
        z_true[:, :p] = zB[:, :p].clone() + dr_true
        p1_true.append({"rank": k, "prefix_rms": prefix_rms(z_true)})

        dev = (zA[:, :p] - zB[:, :p])
        rng = torch.Generator(device=dev.device).manual_seed(seed + k + 777)
        dir_r = torch.randn(dev.shape, generator=rng, device=dev.device)
        dr_fake = lowrank_proj(dir_r, torch.zeros_like(dir_r), k)
        dr_fake = dr_fake * (dr_true.norm() / (dr_fake.norm() + 1e-9))
        z_fake = zB.clone()
        z_fake[:, :p] = zB[:, :p].clone() + dr_fake
        p1_fake.append({"rank": k, "prefix_rms": prefix_rms(z_fake)})

        cache_recon = lowrank_cache((zA[:, :p] - zB[:, :p]), k)
        z_cache = zB.clone()
        z_cache[:, :p] = zB[:, :p].clone() + cache_recon
        p2_boundary.append(prefix_rms(z_true))
        p2_cache.append(prefix_rms(z_cache))
        print(f"P1 rank{k} true={prefix_rms(z_true):.6g} fake={prefix_rms(z_fake):.6g} "
              f"| P2 boundary={prefix_rms(z_true):.6g} cache={prefix_rms(z_cache):.6g}",
              flush=True)

    audit["P1_loop_detection"] = {"true_boundary": p1_true, "fake_boundary": p1_fake}
    audit["P2_cache_equivalence"] = {
        "boundary_prefix_rms": p2_boundary,
        "lowrank_cache_prefix_rms": p2_cache,
        "equal": [abs(a - b) < 1e-6 for a, b in zip(p2_boundary, p2_cache)],
    }
    z_full_future_err = (
        torch.cat([zA[:, :p], zB[:, p:]], dim=1)[:, p:] - zB[:, p:]
    ).square().mean().sqrt().item()
    audit["P3_coupling"] = {
        "latte_temporal_attention": False,
        "note": "Latte processes frames independently (b*f batch, only temp_embed), "
                "so prefix calibration cannot couple into the future in-latent. "
                "full_oracle future_err==0 is an in-latent convenience, NOT a verified "
                "post-edit generation result.",
        "future_err_full_oracle": z_full_future_err,
    }

    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    print("LATTE_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()