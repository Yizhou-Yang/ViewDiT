#!/usr/bin/env python3
"""Protocol Point 1 ablation: does RELAXING prefix protection let the frozen
DiT actually complete the semantic edit (brown bear -> polar bear)?

Ablates pin strength at denoising time (DiT latent), NOT at the decoder.
  --protect=none : no prefix pinning -> all frames free to be edited by prompt
  --protect=half : pin alternate-ish (a cheap middle ground)
  --protect=full : pin all prefix frames (bear49 default baseline)

Because the edit target is a white/light animal, we objectively measure
"whiteness" of the edited region: high brightness AND low chroma (R~G~B).
If relaxing protection makes the edited frames turn white (polar-bear color)
while full protection keeps them brown, then protection suppresses the edit.
If even with NO protection the region stays brown, the frozen DiT cannot do
the semantic edit at all (model capability, not our protection) -- decisive.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXVideoToVideoPipeline, CogVideoXDDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from cog_boundary_moment_transport import capture
from cog_exact_memory import NoUnusedTemporalCache

MODEL = Path('/root/viewdit/weights/CogVideoX-2b')
OUT = Path('/root/viewdit/results/video/cog_semantic_protect_ablate')
CASES = {
    'bear': ['A white polar bear walking on rocky ground, realistic wildlife footage.',
             'A black and white panda walking on rocky ground, realistic wildlife footage.'],
}
FALLBACK_EMBED = '/root/viewdit/results/video/cog_semantic_bear49/embeddings.pt'


def sync():
    torch.cuda.synchronize()


def load_video(clip, frames, height, width):
    root = Path('/root/viewdit/data/v3_validation_data')
    files = sorted(p for p in (root / 'JPEGImages' / clip).glob('*.jpg') if not p.name.startswith('.'))
    assert len(files) >= frames, f'{clip}: only {len(files)} source frames, requested {frames}'
    arrays = [np.asarray(Image.open(p).convert('RGB').resize((width, height)), dtype=np.float32) / 127.5 - 1
              for p in files[:frames]]
    return torch.from_numpy(np.stack(arrays)).permute(3, 0, 1, 2).unsqueeze(0).cuda()


def render(videos, path):
    keys = list(videos)
    frames = []
    for t in range(next(iter(videos.values())).shape[2]):
        panel = Image.new('RGB', (320 * len(keys), 236), 'white')
        draw = ImageDraw.Draw(panel)
        for j, k in enumerate(keys):
            a = videos[k][0, :, t].float().clamp(-1, 1).permute(1, 2, 0)
            im = Image.fromarray(((a + 1) * 127.5).round().byte().numpy()).resize((320, 212))
            panel.paste(im, (320 * j, 24))
            draw.text((320 * j + 4, 4), k + ' f' + str(t), fill='black')
        frames.append(panel)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=125, loop=0)
    select = [0, len(frames) // 2, len(frames) - 2, len(frames) - 1]
    sheet = Image.new('RGB', (frames[0].width, 236 * len(select)), 'white')
    for i, t in enumerate(select):
        sheet.paste(frames[t], (0, i * 236))
    sheet.save(path.with_suffix('.png'))


def whiteness(v):
    """Brightness and whiteness of a video tensor [-1,1] in (1,C,T,H,W).

    Returns (mean_brightness_01, mean_chroma): chroma=mean(|R-G|+|G-B|+|B-R|)/3.
    Polar-bear-white => high brightness, very low chroma.
    """
    a = (v.float().clamp(-1, 1) + 1) / 2
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    bright = (r + g + b) / 3
    chroma = (torch.abs(r - g) + torch.abs(g - b) + torch.abs(b - r)) / 3
    return float(bright.mean()), float(chroma.mean())


@torch.no_grad()
def embeddings(clips, cache):
    if cache.exists():
        return torch.load(cache, map_location='cpu', weights_only=True)
    if Path(FALLBACK_EMBED).exists():
        src = torch.load(FALLBACK_EMBED, map_location='cpu', weights_only=True)
        need = {''} | {p for c in clips for p in CASES[c]}
        assert need <= set(src), f'missing embeddings in fallback: {need - set(src)}'
        torch.save(src, cache)
        return src
    tokenizer = T5Tokenizer.from_pretrained(MODEL / 'tokenizer', local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=torch.float32,
                                             local_files_only=True).cuda().eval()
    result = {}
    for text in [''] + [p for c in clips for p in CASES[c]]:
        tokens = tokenizer(text, padding='max_length', max_length=226, truncation=True,
                           add_special_tokens=True, return_tensors='pt')
        result[text] = encoder(tokens.input_ids.cuda())[0].half().cpu()
        print('EMBED', text, flush=True)
    torch.save(result, cache)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips', default='bear')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--frames', type=int, default=17)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--strength', type=float, default=.6)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--start-latent', type=int, default=4)
    parser.add_argument('--protect', default='full', choices=['none', 'half', 'full'])
    parser.add_argument('--skip-roundtrip', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    clips = args.clips.split(',')
    embed = embeddings(clips, args.out / 'embeddings.pt')
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL / 'vae', torch_dtype=torch.float32,
                                                 local_files_only=True).cuda().eval().requires_grad_(False)
    memory_patch = NoUnusedTemporalCache(vae)
    dit = CogVideoXTransformer3DModel.from_pretrained(MODEL / 'transformer', torch_dtype=torch.float16,
                                                      local_files_only=True).eval().requires_grad_(False)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(MODEL / 'scheduler', local_files_only=True)
    pipe = CogVideoXVideoToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae,
                                         transformer=dit, scheduler=scheduler)
    report = {'complete': False,
              'protocol': vars(args) | {'out': str(args.out), 'seed': 20260921, 'protect_mode': args.protect,
                                        'question': 'Does relaxing DiT prefix protection let frozen model reach polar-bear color?',
                                        'whiteness_metric': 'brightness(0..1 mean of RGB) and chroma(0..1 mean of |RG|+|GB|+|BR|)/3 on edited region. Polar-white: high bright & low chroma.'},
              'rows': []}

    def save(row=None):
        if row is not None:
            report['rows'].append(row)
            print('ROW', json.dumps(row), flush=True)
        p = args.out / 'stats.tmp'
        p.write_text(json.dumps(report, indent=2))
        p.replace(args.out / 'stats.json')

    save()
    for clip in clips:
        x = load_video(clip, args.frames, args.height, args.width)
        sync()
        tic = time.perf_counter()
        z = vae.encode(x).latent_dist.mode()
        sync()
        encode_sec = time.perf_counter() - tic
        del x
        base, _ = capture(vae, z)
        z_current = z.clone()
        latent_frames = z.shape[2]
        assert latent_frames >= 5 and latent_frames % 2 == 1
        assert args.start_latent >= 4 and args.start_latent % 2 == 0 and args.start_latent < latent_frames
        protected_end = 1 + 4 * (args.start_latent - 1)
        for rnd in range(args.rounds):
            prompt = CASES[clip][rnd]
            source = z_current.permute(0, 2, 1, 3, 4).to(torch.float16) * vae.config.scaling_factor
            scheduler.set_timesteps(args.steps, device='cuda')
            used_steps = scheduler.timesteps[args.steps - int(args.steps * args.strength):]
            assert len(used_steps) > 0
            generator = torch.Generator(device='cuda').manual_seed(20260921 + rnd)
            noise = torch.randn(source.shape, device='cuda', dtype=source.dtype, generator=generator)
            initial = scheduler.add_noise(source, noise, used_steps[:1])

            def make_pin(protect, start_latent, src, nz, steps, slen):
                def pin(p, i, t, kw):
                    if protect == 'none':
                        return {'latents': kw['latents']}
                    value = kw['latents']
                    pinned = src if i == slen - 1 else scheduler.add_noise(src, nz, steps[i + 1:i + 2])
                    if protect == 'half':
                        half = start_latent // 2
                        value[:, :half] = pinned[:, :half]
                    else:  # full
                        value[:, :start_latent] = pinned[:, :start_latent]
                    return {'latents': value}
                return pin

            pin = make_pin(args.protect, args.start_latent, source, noise, used_steps, len(used_steps))
            dit.cuda()
            torch.cuda.reset_peak_memory_stats()
            sync()
            tic = time.perf_counter()
            latent = pipe(latents=initial.clone(), prompt_embeds=embed[prompt].cuda(),
                          negative_prompt_embeds=embed[''].cuda(), height=args.height, width=args.width,
                          num_inference_steps=args.steps, strength=args.strength, guidance_scale=6., eta=0.,
                          generator=generator, output_type='latent', callback_on_step_end=pin).frames
            sync()
            dit_sec = time.perf_counter() - tic
            peak = torch.cuda.max_memory_allocated()
            assert torch.isfinite(latent).all(), 'nonfinite DiT'
            dit.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            ze = latent.float().permute(0, 2, 1, 3, 4) / vae.config.scaling_factor
            ze[:, :, :args.start_latent] = z[:, :, :args.start_latent]
            y, _ = capture(vae, ze)
            assert torch.isfinite(y).all()

            edit = y[:, :, protected_end:]
            base_edit = base[:, :, protected_end:]
            bright, chroma = whiteness(edit)
            base_bright, base_chroma = whiteness(base_edit)
            edit_rms = float((edit - base_edit).square().mean().sqrt())
            protected_max = float((y[:, :, :protected_end] - base[:, :, :protected_end]).abs().max())
            row = {'clip': clip, 'round': rnd + 1, 'protect': args.protect, 'prompt': prompt,
                   'dit_seconds': dit_sec, 'dit_peak_allocated_bytes': peak, 'input_encode_seconds': encode_sec,
                   'protected_max': protected_max, 'edit_rms': edit_rms,
                   'edit_brightness': bright, 'base_brightness': base_bright,
                   'edit_chroma': chroma, 'base_chroma': base_chroma,
                   'brightness_gain': bright - base_bright, 'chroma_drop': base_chroma - chroma}
            save(row)
            videos = {'base': base.cpu(), 'out_' + args.protect: y.cpu()}
            torch.save({'videos': videos, 'latent': ze.cpu(), 'source_latent': z.cpu(), 'prompt': prompt},
                       args.out / (clip + '_round' + str(rnd + 1) + '_' + args.protect + '.pt'))
            render(videos, args.out / (clip + '_round' + str(rnd + 1) + '_' + args.protect + '.gif'))
            z_current = ze
            del y, edit, base_edit
            gc.collect()
            torch.cuda.empty_cache()
    memory_patch.close()
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()