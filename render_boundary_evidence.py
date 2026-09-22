#!/usr/bin/env python3
"""Render raw frame and amplified error comparisons without selecting best outcomes."""
from pathlib import Path
import torch
from PIL import Image, ImageDraw
ROOT = Path('/root/viewdit/results/video')
OUT = ROOT / 'boundary_evidence'


def image(tensor):
    x = tensor.detach().float().clamp(-1, 1)
    return Image.fromarray(((x.permute(1, 2, 0) + 1) * 127.5).round().byte().numpy())


def make(name, file, names, error_names=()):
    d = torch.load(file, map_location='cpu', weights_only=True)
    videos = {k: v[0] if v.ndim == 5 else v for k, v in d['videos'].items()}
    keys = [k for k in names if k in videos]
    frames = [4, 5, 8, 9, 12, 13, 16]
    h, w = videos[keys[0]].shape[-2:]
    rows = len(keys) + len(error_names)
    sheet = Image.new('RGB', (160 + len(frames) * w, 36 + rows * (h + 22)), '#ffffff')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), name + ' | 0-based frames | error maps x20', fill='black')
    for j, frame in enumerate(frames):
        draw.text((160 + j*w, 24), f'frame {frame}', fill='black')
    for i, key in enumerate(keys):
        yy = 36 + i*(h+22)
        draw.text((8, yy+10), key, fill='black')
        for j, frame in enumerate(frames):
            sheet.paste(image(videos[key][:, frame]), (160+j*w, yy+22))
    for i, key in enumerate(error_names, start=len(keys)):
        yy = 36 + i*(h+22)
        draw.text((8, yy+10), f'{key}-base x20', fill='black')
        for j, frame in enumerate(frames):
            err = ((videos[key][:, frame]-videos['base'][:, frame]).abs()*20).clamp(0, 1)
            sheet.paste(Image.fromarray((err.permute(1,2,0)*255).round().byte().numpy()), (160+j*w, yy+22))
    sheet.save(OUT / f'{name}.png')
    animation=[]
    for frame in range(17):
        panel = Image.new('RGB', (len(keys)*w, h+44), 'white')
        pen=ImageDraw.Draw(panel)
        for j,key in enumerate(keys):
            pen.text((j*w+4,4),key,fill='black')
            pen.text((j*w+4,20),f'frame {frame}',fill='black')
            panel.paste(image(videos[key][:,frame]),(j*w,44))
        animation.append(panel)
    animation[0].save(OUT/f'{name}.gif',save_all=True,append_images=animation[1:],duration=180,loop=0)
    print(name, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make('cog_bus_geometric',ROOT/'cog_norm_holdout/bus_4_geometric_0.4.pt',['base','raw','replay','pixel_splice','framewise'],['raw','replay'])
    make('cog_boat_color',ROOT/'cog_norm_holdout/boat_2_encoded_color_patch.pt',['base','raw','replay','pixel_splice','framewise'],['raw','replay'])
    make('wan_camel_bridge',ROOT/'wan_transition_budget/camel_latent_geometric_impulse_start13.pt',['base','edit','free_bridge','reencoded_fade','pixel_fade'])
    make('wan_bear_bridge',ROOT/'wan_transition_budget/bear_pixel_color_patch_encoded_start13.pt',['base','edit','free_bridge','reencoded_fade','pixel_fade'])


if __name__ == '__main__':
    main()
