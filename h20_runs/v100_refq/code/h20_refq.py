#!/usr/bin/env python3
"""Reference-free (VBench-style) quality for every generated latent: does NOT compare to full.
Metrics per video (49 frames decoded by the frozen CogVideoX VAE, which is only a measuring tool):
  aesthetic   : LAION aesthetic predictor v1 (linear head on CLIP ViT-L/14 image embeds), mean over frames  (VBench aesthetic_quality)
  clip_text   : CLIP-L text-video similarity x100                                                       (overall consistency proxy)
  subj_cons   : mean cosine of CLIP image embeds to frame 0 and to previous frame (VBench subject consistency uses DINO)
  imaging     : mean Laplacian-variance sharpness (imaging-quality proxy; MUSIQ unavailable w/o cv2)
  motion      : mean abs frame difference (dynamic degree proxy)
  flicker     : mean abs 2nd temporal difference (temporal flickering proxy, lower better)
Usage: COG_MODEL=2b COG_OUT=... COG_LAT=... python3 h20_refq.py --phase fair [--suf _g0 --shard 0 --nshard 4]
"""
import argparse, glob, json, os
from pathlib import Path
import torch

ROOT = Path(os.environ.get('VROOT', '/data/workspace/viewdit'))
MNAME = os.environ.get('COG_MODEL', '2b')
MODEL = ROOT / f'weights/CogVideoX-{MNAME}'
DT = torch.float16 if MNAME == '2b' else torch.bfloat16
OUT = Path(os.environ['COG_OUT'])
LAT = Path(os.environ['COG_LAT'])
import importlib.util as _u; _s=_u.spec_from_file_location('hb', os.path.join(os.path.dirname(os.path.abspath(__file__)),'h20_cog_bench.py')); _m=_u.module_from_spec(_s); _s.loader.exec_module(_m); TEST=_m.TEST


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True); p.add_argument('--suf', default='')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--lat_dirs', nargs='*', default=[])
    a = p.parse_args()
    from diffusers import AutoencoderKLCogVideoX
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    L = OUT / f'refq_{a.phase}{a.suf}.jsonl'
    done = set()
    for fn in glob.glob(str(OUT / f'refq_{a.phase}*.jsonl')):
        for r in map(json.loads, open(fn)):
            done.add((r['prompt'], r['seed'], r['method']))
    gen = [json.loads(l) for fn in glob.glob(str(OUT / f'gen_{a.phase}*.jsonl')) for l in open(fn)]
    items = sorted({(r['prompt'], r['seed'], r['method']) for r in gen})
    items = [it for n, it in enumerate(items) if n % a.nshard == a.shard and it not in done]
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL / 'vae', torch_dtype=DT).cuda().eval().requires_grad_(False)
    vae.enable_tiling()
    cp = ROOT / 'weights/clip-vit-large-patch14'
    clip = CLIPModel.from_pretrained(cp, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(cp)
    head = torch.nn.Linear(768, 1).cuda()
    head.load_state_dict(torch.load(ROOT / 'weights/aesthetic/sa_0_4_vit_l_14_linear.pth', map_location='cuda'))
    with torch.no_grad():
        for i, s, name in items:
            f = LAT / f'p{i:02d}_s{s}' / f'{name}.pt'
            if not f.exists():
                print('missing', f, flush=True); continue
            z = torch.load(f, weights_only=True).to(DT).cuda()
            v = vae.decode(z.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor).sample.float()
            v = ((v.clamp(-1, 1) + 1) / 2)[0]
            vc = v.cpu()
            frames = [Image.fromarray((vc[:, t].permute(1, 2, 0) * 255).round().byte().numpy()) for t in range(vc.shape[1])]
            imf = clip.get_image_features(pixel_values=proc(images=frames, return_tensors='pt')['pixel_values'].half().cuda()).float()
            aest = float(head(imf / imf.norm(dim=-1, keepdim=True)).mean())
            imf = imf / imf.norm(dim=-1, keepdim=True)
            tf = clip.get_text_features(**{k: v_.cuda() for k, v_ in proc(text=[TEST[i]], return_tensors='pt', padding=True, truncation=True).items()}).float()
            tf = tf / tf.norm(dim=-1, keepdim=True)
            gray = v.mean(0)
            lap = gray[:, 1:-1, 1:-1] * 4 - gray[:, :-2, 1:-1] - gray[:, 2:, 1:-1] - gray[:, 1:-1, :-2] - gray[:, 1:-1, 2:]
            r = dict(phase=a.phase, prompt=i, seed=s, method=name, aesthetic=aest, clip_text=float((imf @ tf.T).mean() * 100),
                     subj_cons=float(0.5 * (imf[1:] @ imf[0]).mean() + 0.5 * (imf[1:] * imf[:-1]).sum(-1).mean()),
                     imaging=float(lap.var(dim=(1, 2)).mean()), motion=float((gray[1:] - gray[:-1]).abs().mean()),
                     flicker=float((gray[2:] - 2 * gray[1:-1] + gray[:-2]).abs().mean()))
            with open(L, 'a') as fh:
                fh.write(json.dumps(r) + '\n')
            print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
