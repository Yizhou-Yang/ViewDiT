#!/usr/bin/env python3
"""Development mechanism study, NOT a novel caching engine or SOTA claim.
Existing environment: torch2.6, diffusers0.31, transformers4.45, sentencepiece.
Pure-noise CogVideoX T2V; 17 frames at 480x720; DDIM30 CFG6.
4 fixed disjoint groups; singles calibrate terminal displacement Gram.
All 6 same-cost pairs evaluated, but held-out outcomes never select a policy.
"""
import argparse
import gc
import hashlib
import itertools
import json
import time
import types
from pathlib import Path
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel, AutoencoderKLCogVideoX, CogVideoXDDIMScheduler

CASES = [
    ('cal_water', 'A white swan swims slowly across a calm blue pond, gentle ripples follow the swan, steady camera, realistic wildlife footage.', 2026092301),
    ('test_street', 'A red bus drives slowly along a city street, buildings in the background, steady camera, realistic footage.', 2026092302),
]
GROUPS = [list(range(3+6*i, 9+6*i)) for i in range(4)]
PAIRS = list(itertools.combinations(range(4), 2))

class DevicePipeline(CogVideoXPipeline):
    @property
    def _execution_device(self):
        return self.transformer.device

class Cache:
    def __init__(self, model):
        self.model = model
        self.originals = [b.forward for b in model.transformer_blocks]
        self.step = -1
        self.selected = set()
        self.cache = {}
        self.full_calls = 0
        self.reused_calls = 0
        self.handle = model.register_forward_pre_hook(self.next_step)
        for i, b in enumerate(model.transformer_blocks):
            def forward(block, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, idx=i):
                chosen = idx in self.selected
                reuse = chosen and self.step in self.skip_steps
                if reuse:
                    dh, de = self.cache[idx]
                    self.reused_calls += 1
                    return hidden_states + dh, encoder_hidden_states + de
                out = self.originals[idx](hidden_states, encoder_hidden_states, temb, image_rotary_emb)
                self.full_calls += 1
                if chosen:
                    self.cache[idx] = ((out[0]-hidden_states).detach(), (out[1]-encoder_hidden_states).detach())
                return out
            b.forward = types.MethodType(forward, b)

    def next_step(self, model, inputs):
        self.step += 1

    def reset(self, groups, steps):
        self.cache.clear()
        self.step = -1
        self.selected = set(itertools.chain.from_iterable(GROUPS[g] for g in groups))
        self.skip_steps = set(range(5, 26, 2)) if steps == 30 else set()
        self.full_calls = self.reused_calls = 0

    def close(self):
        self.handle.remove()
        for b, original in zip(self.model.transformer_blocks, self.originals):
            b.forward = original
        self.cache.clear()


def analyze(root):
    outputs = {case: torch.load(root/f'{case}_outputs.pt', map_location='cpu', weights_only=True) for case, _, _ in CASES}
    cal = outputs[CASES[0][0]]
    reference = cal['full'].float()
    delta = torch.stack([(cal[f'single_{i}'].float()-reference).flatten() for i in range(4)])
    gram = delta @ delta.T / delta.shape[1]
    diagonal = [float(gram[i,i]+gram[j,j]) for i,j in PAIRS]
    joint = [float(gram[i,i]+gram[j,j]+2*gram[i,j]) for i,j in PAIRS]
    chosen_diag = PAIRS[min(range(6), key=lambda k: diagonal[k])]
    chosen_joint = PAIRS[min(range(6), key=lambda k: joint[k])]
    report = {'calibration_case': CASES[0][0], 'gram': gram.tolist(), 'pairs': PAIRS, 'diagonal_prediction': diagonal, 'joint_prediction': joint, 'chosen_diagonal': chosen_diag, 'chosen_joint': chosen_joint, 'cases': {}}
    for case, _, _ in CASES:
        out = outputs[case]
        full = out['full'].float()
        def mse(key):
            return float((out[key].float()-full).square().mean())
        errors = {k: mse(k) for k in out}
        oracle = min(PAIRS, key=lambda p: errors[f'pair_{p[0]}_{p[1]}'])
        singles = [out[f'single_{i}'].float()-full for i in range(4)]
        nonlinear = {}
        for i,j in PAIRS:
            actual = out[f'pair_{i}_{j}'].float()-full
            predicted = singles[i]+singles[j]
            nonlinear[f'{i}_{j}'] = float((actual-predicted).norm()/actual.norm().clamp_min(1e-12))
        report['cases'][case] = {'mse_to_full_not_quality': errors, 'oracle_pair_diagnostic_only': oracle, 'diagonal_selected_mse': errors[f'pair_{chosen_diag[0]}_{chosen_diag[1]}'], 'joint_selected_mse': errors[f'pair_{chosen_joint[0]}_{chosen_joint[1]}'], 'linear_superposition_relative_error': nonlinear}
    (root/'analysis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/root/viewdit/weights/CogVideoX-2b'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    stats = {'complete': False, 'protocol': {'cases': CASES, 'groups': GROUPS, 'skip_steps_zero_indexed': list(range(5,26,2)), 'frames':17, 'height':480, 'width':720, 'steps':30, 'cfg':6, 'eta':0, 'source_video':None, 'selection':'calibration prompt singles only; no test access', 'limitations':'one calibration and one held-out development prompt; not a confirmatory benchmark'}, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'gpu':torch.cuda.get_device_name(), 'runs':[]}
    def save():
        tmp = args.out/'stats.tmp'
        tmp.write_text(json.dumps(stats,indent=2))
        tmp.replace(args.out/'stats.json')
    save()
    tokenizer = T5Tokenizer.from_pretrained(args.model/'tokenizer', local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(args.model/'text_encoder', torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    embeds = {}
    for prompt in ['', *[c[1] for c in CASES]]:
        tokens = tokenizer(prompt, padding='max_length', max_length=226, truncation=True, return_tensors='pt')
        embeds[prompt] = encoder(tokens.input_ids.cuda())[0].half().cpu()
    torch.save(embeds, args.out/'embeddings.pt')
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model/'vae', torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False)
    dit = CogVideoXTransformer3DModel.from_pretrained(args.model/'transformer', torch_dtype=torch.float16, local_files_only=True).cuda().eval().requires_grad_(False)
    scheduler = CogVideoXDDIMScheduler.from_pretrained(args.model/'scheduler', local_files_only=True)
    pipe = DevicePipeline(vae=vae,transformer=dit,scheduler=scheduler,tokenizer=None,text_encoder=None)
    pipe.set_progress_bar_config(disable=True)
    cache = Cache(dit)
    specs = [('full', (), 30)] + [(f'single_{i}', (i,),30) for i in range(4)] + [(f'pair_{i}_{j}',(i,j),30) for i,j in PAIRS] + [('fewer26',(),26)]
    for case,prompt,seed in CASES:
        outputs = {}
        initial = torch.randn((1,5,16,60,90), generator=torch.Generator(device='cuda').manual_seed(seed),device='cuda',dtype=torch.float16)
        torch.save(initial.cpu(), args.out/f'{case}_initial.pt')
        for name, groups, steps in specs:
            cache.reset(groups,steps)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = pipe(latents=initial.clone(), prompt_embeds=embeds[prompt].cuda(),negative_prompt_embeds=embeds[''].cuda(),height=480,width=720,num_frames=17,num_inference_steps=steps,guidance_scale=6.,eta=0.,generator=torch.Generator(device='cuda').manual_seed(seed),output_type='latent').frames
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            assert torch.isfinite(result).all()
            assert cache.step+1 == steps
            outputs[name] = result.cpu()
            row = {'case':case,'method':name,'seconds_denoising_only':elapsed,'actual_block_calls':cache.full_calls,'reused_block_calls':cache.reused_calls,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'finite':True}
            stats['runs'].append(row)
            torch.save(outputs,args.out/f'{case}_outputs.pt')
            save()
            print(json.dumps(row),flush=True)
            del result
        gc.collect()
    cache.close()
    analyze(args.out)
    stats['complete'] = True
    save()

if __name__ == '__main__':
    main()
