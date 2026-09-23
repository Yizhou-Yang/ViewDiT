#!/usr/bin/env python3
"""Real CogVideoX sampling pool. Standard assignment baseline, NOT new method.
Existing torch2.6 diffusers0.31 transformers4.45 scipy1.14 numpy Pillow.
"""
import argparse
import gc
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXVideoToVideoPipeline, CogVideoXDDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from cog_exact_memory import NoUnusedTemporalCache

CASES = {
    'bear': ['A white polar bear walking on rocky ground, realistic wildlife footage.', 'A black and white panda walking on rocky ground, realistic wildlife footage.'],
    'blackswan': ['A white swan swimming on a pond, realistic wildlife footage.', 'A pink swan swimming on a pond, realistic wildlife footage.'],
    'bus': ['A bright red bus driving on a city street, realistic video.', 'A bright yellow bus driving on a city street, realistic video.'],
    'boat': ['A red sailboat sailing on water, realistic video.', 'A yellow sailboat sailing on water, realistic video.'],
}


def load_source(root, clip, h, w):
    files = sorted(p for p in (root / 'JPEGImages' / clip).glob('*.jpg') if not p.name.startswith('.'))[:17]
    assert len(files) == 17
    x = np.stack([np.asarray(Image.open(p).convert('RGB').resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32) / 127.5 - 1 for p in files])
    return torch.from_numpy(x).permute(3, 0, 1, 2)[None].cuda()


def panel(source, base, pools, path):
    columns = [('source', source[0]), ('VAE reconstruction', base[0]), ('branch A seed0', pools[0][0]), ('branch B seed0', pools[1][0])]
    canvas = Image.new('RGB', (4*336, 3*216), 'white')
    draw = ImageDraw.Draw(canvas)
    for r, t in enumerate([0, 8, 16]):
        for c, (name, video) in enumerate(columns):
            a = ((video[:, t].float().clamp(-1, 1).permute(1, 2, 0) + 1)*127.5).round().byte().numpy()
            canvas.paste(Image.fromarray(a).resize((336, 192)), (c*336, r*216+24))
            draw.text((c*336+3, r*216+3), name + f' frame{t}', fill='black')
    canvas.save(path)


def analyze(pools):
    a, b = [v.float() for v in pools]
    n, channels, frames, h, w = a.shape
    border = torch.ones(h, w, dtype=torch.bool)
    border[int(.2*h):int(.8*h), int(.2*w):int(.8*w)] = False
    def costs(time_ids, region):
        aa = a[:, :, time_ids, :, :][..., region].flatten(1)
        bb = b[:, :, time_ids, :, :][..., region].flatten(1)
        return (aa[:, None] - bb[None]).square().mean(-1).numpy()
    train = costs(list(range(0, frames, 2)), border)
    test = costs(list(range(1, frames, 2)), border)
    full = costs(list(range(1, frames, 2)), torch.ones_like(border))
    center = costs(list(range(1, frames, 2)), ~border)
    ii, jj = linear_sum_assignment(train)
    assert (ii == np.arange(n)).all() and sorted(jj.tolist()) == list(range(n))
    perms = np.array(list(itertools.permutations(range(n))))
    random_costs = test[np.arange(n)[None], perms].mean(1)
    paired = float(test[ii, jj].mean())
    same = float(test.trace()/n)
    random = float(test.mean())
    def diversity(v):
        vec = v.flatten(1)
        return float(torch.pdist(vec).square().mean()/vec.shape[1])
    return {'same_seed_odd_border_mse': same, 'random_permutation_expected_odd_border_mse': random, 'matched_odd_border_mse': paired, 'reduction_vs_same': 1-paired/max(same, 1e-12), 'reduction_vs_random': 1-paired/max(random, 1e-12), 'permutation_tail_fraction_diagnostic_not_scene_p': float((random_costs <= paired+1e-12).mean()), 'assignment': jj.tolist(), 'same_full_mse': float(full.trace()/n), 'matched_full_mse': float(full[ii, jj].mean()), 'same_center_mse': float(center.trace()/n), 'matched_center_mse': float(center[ii, jj].mean()), 'within_branch_diversity': [diversity(a), diversity(b)], 'marginal_multiset_preserved': True, 'even_border_cost_matrix': train.tolist(), 'odd_border_cost_matrix': test.tolist()}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/root/viewdit/weights/CogVideoX-2b'))
    p.add_argument('--data', type=Path, default=Path('/root/viewdit/data/v3_validation_data'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20260922)
    report = {'complete': False, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'protocol': {'h':128,'w':224,'frames':17,'steps':50,'strength':.75,'cfg':6.,'seeds':list(range(20261001,20261009)),'prompts':CASES,'method':'untrained V2V with DDIM eta0; no prefix pins','limits':'development clips; low resolution; no semantic success metric; border proxy not mask; assignment is standard baseline'}, 'clips': {}, 'sampling': []}
    def save():
        tmp = args.out/'stats.tmp'
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.out/'stats.json')
    save()
    tok = T5Tokenizer.from_pretrained(args.model/'tokenizer', local_files_only=True)
    enc = T5EncoderModel.from_pretrained(args.model/'text_encoder', torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    embeds = {}
    for text in ['']+[s for prompts in CASES.values() for s in prompts]:
        inputs = tok(text, padding='max_length', max_length=226, truncation=True, add_special_tokens=True, return_tensors='pt')
        embeds[text] = enc(inputs.input_ids.cuda())[0].half().cpu()
        assert torch.isfinite(embeds[text]).all()
    torch.save(embeds, args.out/'embeddings.pt')
    del tok, enc
    gc.collect(); torch.cuda.empty_cache()
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model/'vae', torch_dtype=torch.float32, local_files_only=True).cuda().eval().requires_grad_(False)
    patch = NoUnusedTemporalCache(vae)
    sources, bases, latents = {}, {}, {}
    for clip in CASES:
        x = load_source(args.data, clip, 128, 224)
        z = vae.encode(x).latent_dist.mode()
        base = vae.decode(z).sample
        assert torch.isfinite(z).all() and torch.isfinite(base).all()
        sources[clip], bases[clip], latents[clip] = x.cpu(), base.cpu(), z.cpu()
    vae.cpu(); gc.collect(); torch.cuda.empty_cache()
    dit = CogVideoXTransformer3DModel.from_pretrained(args.model/'transformer', torch_dtype=torch.float16, local_files_only=True).cuda().eval().requires_grad_(False)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(args.model/'scheduler', local_files_only=True)
    class ExplicitDevicePipeline(CogVideoXVideoToVideoPipeline):
        @property
        def _execution_device(self):
            return self.transformer.device
    pipe = ExplicitDevicePipeline(vae=vae, transformer=dit, scheduler=scheduler, tokenizer=None, text_encoder=None)
    assert pipe._execution_device.type == 'cuda'
    pipe.set_progress_bar_config(disable=True)
    all_latents = {}
    for clip, prompts in CASES.items():
        source = latents[clip].permute(0, 2, 1, 3, 4).cuda().half()*vae.config.scaling_factor
        branches = []
        for branch, prompt in enumerate(prompts):
            outputs = []
            for seed in report['protocol']['seeds']:
                scheduler.set_timesteps(50, device='cuda')
                effective = scheduler.timesteps[50-int(50*.75):]
                gen = torch.Generator(device='cuda').manual_seed(seed)
                noise = torch.randn(source.shape, device='cuda', dtype=source.dtype, generator=gen)
                initial = scheduler.add_noise(source, noise, effective[:1])
                torch.cuda.synchronize(); start = time.perf_counter()
                result = pipe(latents=initial, prompt_embeds=embeds[prompt].cuda(), negative_prompt_embeds=embeds[''].cuda(), height=128, width=224, num_inference_steps=50, strength=.75, guidance_scale=6., eta=0., generator=gen, output_type='latent').frames
                torch.cuda.synchronize()
                assert torch.isfinite(result).all()
                outputs.append(result.cpu())
                row = {'clip':clip,'branch':branch,'seed':seed,'effective_steps':len(effective),'seconds':time.perf_counter()-start}
                report['sampling'].append(row); save()
                print(json.dumps(row), flush=True)
            branches.append(torch.cat(outputs))
        all_latents[clip] = branches
        torch.save({'branches':branches,'source':latents[clip]}, args.out/f'{clip}_latents.pt')
    dit.cpu(); gc.collect(); torch.cuda.empty_cache(); vae.cuda()
    for clip, branches in all_latents.items():
        pools = []
        for zs in branches:
            videos = []
            for z in zs:
                result = vae.decode(z[None].cuda().float().permute(0,2,1,3,4)/vae.config.scaling_factor).sample
                assert torch.isfinite(result).all()
                videos.append(result.cpu())
            pools.append(torch.cat(videos))
        torch.save({'pools':pools,'source':sources[clip],'base':bases[clip]}, args.out/f'{clip}_videos.pt')
        report['clips'][clip] = analyze(pools)
        panel(sources[clip], bases[clip], pools, args.out/f'{clip}_fixed.png')
        save(); print(clip, json.dumps(report['clips'][clip]), flush=True)
    report['complete'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
