#!/usr/bin/env python3
"""Development-only strength diagnostic; not a proposed editing method."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXVideoToVideoPipeline, CogVideoXDDIMScheduler
from cog_exact_memory import NoUnusedTemporalCache

PROMPTS = ['A bright red bus driving on a city street, realistic video.', 'A bright yellow bus driving on a city street, realistic video.']

class ExplicitDevicePipeline(CogVideoXVideoToVideoPipeline):
    @property
    def _execution_device(self):
        return self.transformer.device

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/root/viewdit/weights/CogVideoX-2b'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = {'complete': False, 'purpose': 'development strength screen, no novelty or significance claim', 'protocol': {'clip': 'bus', 'strengths': [.35, .5], 'reference_strength': .75, 'seed': 20261001, 'steps': 50, 'cfg': 6, 'eta': 0, 'frames': 17, 'height': 480, 'width': 720}, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'samples': []}
    def save():
        tmp = args.out / 'stats.tmp'
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.out / 'stats.json')
    save()
    embeddings = torch.load(args.input / 'embeddings.pt', weights_only=True, map_location='cpu')
    source_z = torch.load(args.input / 'bus_latents.pt', weights_only=True, map_location='cpu')['source']
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model / 'vae', torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False)
    patch = NoUnusedTemporalCache(vae)
    dit = CogVideoXTransformer3DModel.from_pretrained(args.model / 'transformer', torch_dtype=torch.float16, local_files_only=True).cuda().eval().requires_grad_(False)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(args.model / 'scheduler', local_files_only=True)
    pipe = ExplicitDevicePipeline(vae=vae, transformer=dit, scheduler=scheduler, tokenizer=None, text_encoder=None)
    pipe.set_progress_bar_config(disable=True)
    assert pipe._execution_device.type == 'cuda'
    source = source_z.permute(0, 2, 1, 3, 4).cuda().half() * vae.config.scaling_factor
    latents = []
    for strength in [.35, .5]:
        for branch, prompt in enumerate(PROMPTS):
            scheduler.set_timesteps(50, device='cuda')
            effective = scheduler.timesteps[50-int(50*strength):]
            gen = torch.Generator(device='cuda').manual_seed(20261001)
            noise = torch.randn(source.shape, device='cuda', dtype=source.dtype, generator=gen)
            initial = scheduler.add_noise(source, noise, effective[:1])
            torch.cuda.synchronize()
            start = time.perf_counter()
            z = pipe(latents=initial, prompt_embeds=embeddings[prompt].cuda(), negative_prompt_embeds=embeddings[''].cuda(), height=480, width=720, num_inference_steps=50, strength=strength, guidance_scale=6., eta=0., generator=gen, output_type='latent').frames
            torch.cuda.synchronize()
            assert torch.isfinite(z).all()
            row = {'strength': strength, 'branch': branch, 'effective_steps': len(effective), 'seconds': time.perf_counter()-start}
            latents.append(z.cpu())
            report['samples'].append(row)
            save()
            print(json.dumps(row), flush=True)
    torch.save(latents, args.out / 'latents.pt')
    dit.cpu()
    del pipe, dit, source
    gc.collect()
    torch.cuda.empty_cache()
    vae.cuda()
    videos = []
    for z in latents:
        video = vae.decode(z.cuda().float().permute(0,2,1,3,4)/vae.config.scaling_factor).sample.cpu()
        assert torch.isfinite(video).all()
        videos.append(video)
    torch.save(videos, args.out / 'videos.pt')
    reference = torch.load(args.input / 'bus_videos.pt', weights_only=True, map_location='cpu')
    source_video = reference['source']
    for row, video in zip(report['samples'], videos):
        row['source_mse_not_edit_success'] = float((video-source_video).square().mean())
        row['finite'] = bool(torch.isfinite(video).all())
    columns = [('source', source_video[0])]
    columns += [(f"s{row['strength']} {'red' if row['branch']==0 else 'yellow'}", video[0]) for row, video in zip(report['samples'], videos)]
    columns += [('s0.75 red', reference['pools'][0][0]), ('s0.75 yellow', reference['pools'][1][0])]
    canvas = Image.new('RGB', (len(columns)*280, 3*185), 'white')
    draw = ImageDraw.Draw(canvas)
    for r, t in enumerate([0,8,16]):
        for c, (name, video) in enumerate(columns):
            image = ((video[:,t].float().clamp(-1,1).permute(1,2,0)+1)*127.5).round().byte().numpy()
            canvas.paste(Image.fromarray(image).resize((280,160)), (c*280,r*185+25))
            draw.text((c*280+3,r*185+3), name+f' frame{t}', fill='black')
    canvas.save(args.out / 'strength_fixed.png')
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    main()
