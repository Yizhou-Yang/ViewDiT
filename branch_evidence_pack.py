#!/usr/bin/env python3
"""Read existing sampled pools, export every seed at frame8 without selection."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    report = {'purpose':'diagnostic only; source MSE is not edit correctness', 'clips':{}}
    for f in sorted(args.root.glob('*_videos.pt')):
        d = torch.load(f, weights_only=True, map_location='cpu')
        pools, source, base = d['pools'], d['source'], d['base']
        n = pools[0].shape[0]
        panel = Image.new('RGB', (4*280, ((n+1)//2)*185), 'white')
        draw = ImageDraw.Draw(panel)
        for b in range(2):
            for i, video in enumerate(pools[b]):
                a = ((video[:,8].float().clamp(-1,1).permute(1,2,0)+1)*127.5).round().byte().numpy()
                x, y = (i%2*2+b)*280, (i//2)*185
                panel.paste(Image.fromarray(a).resize((280,160)), (x,y+25))
                draw.text((x+3,y+3), f'branch{b} seed{i} frame8', fill='black')
        clip = f.name.replace('_videos.pt','')
        panel.save(args.root/f'{clip}_all_seeds.png')
        report['clips'][clip] = {'samples':n*2, 'vae_source_mse':float((base-source).square().mean()), 'branch_source_mse':[float((v-source).square().mean()) for v in pools], 'finite':[bool(torch.isfinite(v).all()) for v in pools], 'shape':[list(v.shape) for v in pools]}
    (args.root/'evidence.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
