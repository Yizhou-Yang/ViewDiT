#!/usr/bin/env python3
"""Decode selected pilot outputs; paired fidelity is not absolute quality."""
import argparse
import json
import time
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKLCogVideoX
from cog_exact_memory import NoUnusedTemporalCache

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/root/viewdit/weights/CogVideoX-2b'))
    args = p.parse_args()
    stats = json.loads((args.root/'stats.json').read_text())
    assert stats['complete']
    analysis = json.loads((args.root/'analysis.json').read_text())
    torch.set_num_threads(2)
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model/'vae',torch_dtype=torch.float32,local_files_only=True).cuda().eval().requires_grad_(False)
    patch = NoUnusedTemporalCache(vae)
    report = {'complete':False,'note':'clamped RGB PSNR to full, temporal-delta error, and visual panels are fidelity checks, not FID/VBench quality','cases':{}}
    for case in analysis['cases']:
        outputs = torch.load(args.root/f'{case}_outputs.pt',map_location='cpu',weights_only=True)
        names = ['full','fewer26']
        choices = {'diagonal':analysis['chosen_diagonal'],'joint':analysis['chosen_joint'],'oracle_diagnostic':analysis['cases'][case]['oracle_pair_diagnostic_only']}
        for pair in choices.values():
            key = f'pair_{pair[0]}_{pair[1]}'
            if key not in names:
                names.append(key)
        forecast_path = args.root/'forecast_baseline'/f'{case}_outputs.pt'
        if forecast_path.exists():
            forecast = torch.load(forecast_path, map_location='cpu', weights_only=True)
            outputs.update(forecast)
            names.extend(forecast.keys())
        videos = {}
        rows = {}
        for name in names:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            v = vae.decode(outputs[name].float().cuda().permute(0,2,1,3,4)/vae.config.scaling_factor).sample
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            assert torch.isfinite(v).all()
            video = ((v.cpu().clamp(-1,1)+1)/2)[0]
            del v
            videos[name] = video
            reference = videos['full']
            mse = float((video-reference).square().mean())
            rows[name] = {'decode_seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'psnr_to_full': None if mse==0 else float(-10*torch.log10(torch.tensor(mse))), 'rgb_mse_to_full':mse,'temporal_difference_error':float(((video[:,1:]-video[:,:-1])-(reference[:,1:]-reference[:,:-1])).square().mean()),'finite':True}
            frames = [Image.fromarray((video[:,t].permute(1,2,0)*255).round().byte().numpy()).resize((360,240)) for t in range(video.shape[1])]
            frames[0].save(args.root/f'{case}_{name}.gif',save_all=True,append_images=frames[1:],duration=125,loop=0)
            print(case,name,rows[name],flush=True)
        torch.save(videos,args.root/f'{case}_decoded.pt')
        canvas = Image.new('RGB',(360*len(names),265*3),'white')
        draw = ImageDraw.Draw(canvas)
        for c,name in enumerate(names):
            for r,t in enumerate([0,8,16]):
                im = Image.fromarray((videos[name][:,t].permute(1,2,0)*255).round().byte().numpy()).resize((360,240))
                canvas.paste(im,(360*c,265*r+25))
                draw.text((360*c+3,265*r+3),f'{name} frame{t}',fill='black')
        canvas.save(args.root/f'{case}_comparison.png')
        report['cases'][case] = {'selection':choices,'metrics':rows}
        (args.root/'decoded_stats.json').write_text(json.dumps(report,indent=2))
    report['complete'] = True
    (args.root/'decoded_stats.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
