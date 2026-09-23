#!/usr/bin/env python3
"""Decode saved strength diagnostic latents in a fresh process, no resampling."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKLCogVideoX
from cog_exact_memory import NoUnusedTemporalCache

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/root/viewdit/weights/CogVideoX-2b'))
    args = p.parse_args()
    torch.set_num_threads(2)
    report = json.loads((args.out/'stats.json').read_text())
    assert not report['complete']
    zs = torch.load(args.out/'latents.pt', weights_only=True, map_location='cpu')
    assert len(zs) == len(report['samples']) == 4
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model/'vae', torch_dtype=torch.float32, local_files_only=True).cuda().eval().requires_grad_(False)
    patch = NoUnusedTemporalCache(vae)
    videos = []
    for i, z in enumerate(zs):
        result = vae.decode(z.cuda().float().permute(0,2,1,3,4)/vae.config.scaling_factor).sample
        assert torch.isfinite(result).all()
        videos.append(result.cpu())
        del result
        torch.cuda.empty_cache()
        print('decoded', i, flush=True)
    torch.save(videos, args.out/'videos.pt')
    reference = torch.load(args.input/'bus_videos.pt', weights_only=True, map_location='cpu')
    source = reference['source']
    columns = [('source', source[0])]
    for row, video in zip(report['samples'], videos):
        row['source_mse_not_edit_success'] = float((video-source).square().mean())
        row['finite'] = bool(torch.isfinite(video).all())
        columns.append((f"s{row['strength']} {'red' if row['branch']==0 else 'yellow'}", video[0]))
    columns += [('s0.75 red', reference['pools'][0][0]), ('s0.75 yellow', reference['pools'][1][0])]
    canvas = Image.new('RGB', (len(columns)*280,3*185), 'white')
    draw = ImageDraw.Draw(canvas)
    for r,t in enumerate([0,8,16]):
        for c,(name,video) in enumerate(columns):
            image = ((video[:,t].clamp(-1,1).permute(1,2,0)+1)*127.5).round().byte().numpy()
            canvas.paste(Image.fromarray(image).resize((280,160)), (c*280,r*185+25))
            draw.text((c*280+3,r*185+3), name+f' frame{t}', fill='black')
    canvas.save(args.out/'strength_fixed.png')
    report['decode_recovery'] = 'fresh FP32 VAE process; expandable_segments allocator; unchanged saved latents and VAE arithmetic; initial inline decoding OOM'
    report['complete'] = True
    tmp = args.out/'stats.tmp'
    tmp.write_text(json.dumps(report,indent=2))
    tmp.replace(args.out/'stats.json')
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    main()
