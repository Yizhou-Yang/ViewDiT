#!/usr/bin/env python3
"""H20 study: does rollout-aware (on-policy, DAgger-style) fitting of a block-residual
cache corrector beat off-policy fitting at equal parameters and equal data size?

Executor: residual reuse on selected CogVideoX blocks at scheduled steps.
Corrector (video stream only): dh_pred = dh_cached + (h_now - h_cached_input) @ W_block,
closed-form ridge. Arms share the SAME form and parameter count; they differ only in the
state distribution of the fitting data.
  ridge_off    : teacher trajectory states, calibration seed A (8 runs)
  ridge_off2x  : teacher states, seeds A+B (16 runs)  <- equal-data control
  ridge_dagger : teacher seed A + rollout under W_off with seed B (16 runs)
  ridge_dagger2: + rollout under W_dagger with seed A (24 runs, exploratory)
Baselines: full30, step-matched DDIM, zero-order reuse, first-order extrapolation.
Test prompts are disjoint from calibration/validation and never used for selection.
Not an official reproduction of any published cache method.
"""
import argparse, gc, hashlib, json, os, time, types
from pathlib import Path
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel, AutoencoderKLCogVideoX, CogVideoXDDIMScheduler

ROOT = Path('/data/viewdit')
MODEL = ROOT / 'weights/CogVideoX-2b'
H, W_, F, STEPS, CFG = 480, 720, 17, 30, 6.0
CAL = [
    'A golden retriever runs across a grassy park chasing a ball, sunny afternoon, realistic footage.',
    'Waves crash against black rocks on a rugged coastline at sunset, cinematic, realistic.',
    'A chef slices fresh vegetables on a wooden cutting board in a bright kitchen, close-up.',
    'A hot air balloon floats slowly over green hills in the early morning mist.',
    'A young woman walks through a busy night market with glowing lanterns, handheld camera.',
    'A hummingbird hovers near a red flower, slow motion, shallow depth of field.',
    'Snow falls gently on a quiet pine forest, a small wooden cabin with warm lights.',
    'A vintage car drives along a desert highway, dust trailing behind, wide shot.',
]
VAL = [
    'A sailboat glides across a sparkling blue lake with mountains in the background.',
    'An old man plays an acoustic guitar on a park bench, autumn leaves falling.',
]
TEST = [
    'A white swan swims slowly across a calm blue pond, gentle ripples follow the swan, steady camera, realistic wildlife footage.',
    'A red bus drives slowly along a city street, buildings in the background, steady camera, realistic footage.',
    'A brown bear walks along a river bank in a dense forest, realistic wildlife documentary.',
    'A panda eats bamboo while sitting in a lush green bamboo forest, close-up.',
    'Fireworks explode over a city skyline at night, reflections in the river.',
    'A cat stretches and yawns on a sunny windowsill, soft natural light.',
    'A surfer rides a large ocean wave, spray in the air, aerial drone shot.',
    'A steaming cup of coffee on a cafe table, rain drops on the window behind.',
    'A horse gallops across an open meadow at golden hour, dramatic lighting.',
    'A timelapse of clouds moving over a snowy mountain peak, blue sky.',
    'A child blows soap bubbles in a garden, bubbles drift in the breeze.',
    'A train crosses a long stone bridge over a valley, steam rising, wide shot.',
    'Colorful koi fish swim in a clear pond with lily pads, top-down view.',
    'A dancer spins on a stage under a single spotlight, dark background.',
    'Autumn leaves swirl in the wind on a cobblestone street in an old town.',
    'A robot arm assembles parts on a factory production line, industrial lighting.',
]
SEED_A, SEED_B = 1000, 2000
TEST_SEEDS = [3000, 4000]
BUDGETS = {  # name: (blocks, refresh interval)
    'B1': (list(range(15, 27)), 2),
    'B2': (list(range(3, 27)), 2),
    'B3': (list(range(3, 27)), 3),
}

def reuse_steps(k, lo=4, hi=26):
    return {s for s in range(lo, hi + 1) if (s - lo) % k != 0}

def matched_steps(budget):
    blocks, k = BUDGETS[budget]
    calls = 30 * 30 - len(blocks) * len(reuse_steps(k))
    return round(calls / 30)

class DevicePipeline(CogVideoXPipeline):
    @property
    def _execution_device(self):
        return self.transformer.device

class Executor:
    def __init__(self, model):
        self.model = model
        self.orig = [b.forward for b in model.transformer_blocks]
        model.register_forward_pre_hook(self._tick)
        self.configure()
        for i, b in enumerate(model.transformer_blocks):
            def fwd(block, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, idx=i):
                return self._forward(idx, hidden_states, encoder_hidden_states, temb, image_rotary_emb)
            b.forward = types.MethodType(fwd, b)

    def _tick(self, *_):
        self.step += 1

    def configure(self, blocks=(), k=2, mode='zero', W=None, collect=None, win=(4, 26)):
        self.blocks = set(blocks)
        self.reuse = reuse_steps(k, *win) if blocks else set()
        self.mode, self.W, self.collect = mode, W, collect
        self.mem = {}
        self.step = -1
        self.full_calls = self.reused_calls = 0

    FEATS = 'aug'
    NB = 3

    def bucket(self):
        return min(self.NB - 1, max(0, (self.step - 4) * self.NB // 23))

    def feat(self, h, m):
        C = h.shape[-1]
        x = (h - m['h']).float().reshape(-1, C)
        if self.FEATS == 'aug':
            x = torch.cat([x, m['dh'].float().reshape(-1, C)], 1)
        return x

    def _forward(self, idx, h, e, temb, rope):
        if idx not in self.blocks or self.step not in self.reuse:
            out = self.orig[idx](h, e, temb, rope)
            self.full_calls += 1
            if idx in self.blocks:
                old = self.mem.get(idx)
                self.mem[idx] = dict(h=h.detach(), dh=(out[0] - h).detach(), de=(out[1] - e).detach(), s=self.step,
                                     pdh=None if old is None else old['dh'], pde=None if old is None else old['de'],
                                     ps=None if old is None else old['s'])
            return out
        m = self.mem[idx]
        dh, de = m['dh'], m['de']
        if self.mode == 'hiboost':
            B_, N_, C_ = dh.shape
            g = dh.float().reshape(B_, 5, 30, 45, C_)
            G = torch.fft.rfft2(g, dim=(2, 3))
            fy = torch.fft.fftfreq(30, device=g.device).abs()[:, None]
            fx = torch.fft.rfftfreq(45, device=g.device)[None, :]
            hi = ((fy ** 2 + fx ** 2).sqrt() > 0.15).to(G.dtype)[None, None, :, :, None]
            g = torch.fft.irfft2(G * hi, s=(30, 45), dim=(2, 3))
            dh = dh + ((float(self.W) - 1) * g).reshape(B_, N_, C_).to(dh.dtype)
        elif self.mode in ('damped', 'freq_hi', 'freq_lo') and m['pdh'] is not None:
            r = (self.step - m['s']) / (m['s'] - m['ps'])
            ddh = r * (dh - m['pdh'])
            if self.mode == 'damped':
                dh = dh + float(self.W) * ddh
            else:
                B_, N_, C_ = ddh.shape
                g = ddh.float().reshape(B_, 5, 30, 45, C_)
                G = torch.fft.rfft2(g, dim=(2, 3))
                fy = torch.fft.fftfreq(30, device=g.device).abs()[:, None]
                fx = torch.fft.rfftfreq(45, device=g.device)[None, :]
                low = ((fy ** 2 + fx ** 2).sqrt() <= float(self.W)).to(G.dtype)[None, None, :, :, None]
                keep = (1 - low) if self.mode == 'freq_hi' else low
                g = torch.fft.irfft2(G * keep, s=(30, 45), dim=(2, 3))
                dh = dh + g.reshape(B_, N_, C_).to(dh.dtype)
        elif self.mode == 'linear' and m['pdh'] is not None:
            r = (self.step - m['s']) / (m['s'] - m['ps'])
            dh = dh + r * (dh - m['pdh'])
            de = de + r * (de - m['pde'])
        elif self.mode.startswith('oracle'):
            true = self.orig[idx](h, e, temb, rope)
            lam = float(self.W)
            if self.mode in ('oracle_dh', 'oracle_both'):
                dh = dh + lam * ((true[0] - h) - dh)
            if self.mode in ('oracle_de', 'oracle_both'):
                de = de + lam * ((true[1] - e) - de)
        elif self.mode == 'ridge' and self.W is not None:
            x = self.feat(h, m)
            dh = dh + (x @ self.W[(idx, self.bucket())]).to(h.dtype).reshape(dh.shape)
        if self.collect is not None:
            true = self.orig[idx](h, e, temb, rope)
            x = self.feat(h, m)
            y = ((true[0] - h) - m['dh']).float().reshape(-1, h.shape[-1])
            c = self.collect.setdefault((idx, self.bucket()), {'xx': 0., 'xy': 0., 'yy': 0., 'n': 0})
            c['xx'] = c['xx'] + x.T @ x
            c['xy'] = c['xy'] + x.T @ y
            c['yy'] = c['yy'] + float(y.square().sum())
            c['n'] += x.shape[0]
            if self.collect.get('_teacher'):
                return true
        self.reused_calls += 1
        return h + dh, e + de

def merge(*stats):
    out = {}
    for s in stats:
        for k, v in s.items():
            if isinstance(k, str):
                continue
            o = out.setdefault(k, {'xx': 0., 'xy': 0., 'yy': 0., 'n': 0})
            for f in o:
                o[f] = o[f] + v[f]
    return out

def solve(stats, alpha):
    W = {}
    for k, v in stats.items():
        xx = v['xx']
        lam = alpha * torch.trace(xx) / xx.shape[0]
        W[k] = torch.linalg.solve(xx + lam * torch.eye(xx.shape[0], device=xx.device), v['xy']).half().float()
    return W

class Runner:
    def __init__(self, out):
        self.out = out
        torch.set_num_threads(8)
        self.embeds = torch.load(ROOT / 'results/embeddings.pt', map_location='cpu', weights_only=True)
        vae = AutoencoderKLCogVideoX.from_pretrained(MODEL / 'vae', torch_dtype=torch.float32).eval()
        self.dit = CogVideoXTransformer3DModel.from_pretrained(MODEL / 'transformer', torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
        sch = CogVideoXDDIMScheduler.from_pretrained(MODEL / 'scheduler')
        self.pipe = DevicePipeline(vae=vae, transformer=self.dit, scheduler=sch, tokenizer=None, text_encoder=None)
        self.pipe.set_progress_bar_config(disable=True)
        self.ex = Executor(self.dit)

    @torch.no_grad()
    def run(self, prompt, seed, steps=STEPS, perturb=0., **cfg):
        self.ex.configure(**cfg)
        init = torch.randn((1, 5, 16, 60, 90), generator=torch.Generator(device='cuda').manual_seed(seed), device='cuda', dtype=torch.float16)
        if perturb:
            eps = torch.randn(init.shape, generator=torch.Generator(device='cuda').manual_seed(seed + 7), device='cuda', dtype=torch.float32)
            init = ((init.float() + perturb * eps) / (1 + perturb ** 2) ** .5).half()
        torch.cuda.synchronize(); t = time.perf_counter()
        z = self.pipe(latents=init, prompt_embeds=self.embeds[prompt].cuda(), negative_prompt_embeds=self.embeds[''].cuda(),
                      height=H, width=W_, num_frames=F, num_inference_steps=steps, guidance_scale=CFG, eta=0.,
                      generator=torch.Generator(device='cuda').manual_seed(seed), output_type='latent').frames
        torch.cuda.synchronize(); sec = time.perf_counter() - t
        assert torch.isfinite(z).all() and self.ex.step + 1 == steps
        return z, dict(seconds=sec, full_calls=self.ex.full_calls, reused_calls=self.ex.reused_calls)

def log(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

def cmd_embed(a):
    tok = T5Tokenizer.from_pretrained(MODEL / 'tokenizer')
    enc = T5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=torch.float32).cuda().eval()
    emb = {}
    with torch.no_grad():
        for p in ['', *CAL, *VAL, *TEST]:
            ids = tok(p, padding='max_length', max_length=226, truncation=True, return_tensors='pt').input_ids.cuda()
            emb[p] = enc(ids)[0].half().cpu()
    (ROOT / 'results').mkdir(parents=True, exist_ok=True)
    torch.save(emb, ROOT / 'results/embeddings.pt')
    print('embeddings', len(emb))

def cmd_fit(a):
    out = ROOT / f'results/fit_{a.budget}'; out.mkdir(parents=True, exist_ok=True)
    R = Runner(out); blocks, k = BUDGETS[a.budget]; L = out / 'fit_log.jsonl'
    def collect(seed, teacher, W=None):
        st = {'_teacher': teacher}
        for i, p in enumerate(CAL):
            _, info = R.run(p, seed, blocks=blocks, k=k, mode='ridge' if W else 'zero', W=W, collect=st)
            log(L, dict(stage='collect', teacher=teacher, seed=seed, prompt=i, **info))
        return {kk: v for kk, v in st.items() if not isinstance(kk, str)}
    sA = collect(SEED_A, True)
    sB = collect(SEED_B, True)
    arms = {}
    val_full = {}
    for i, p in enumerate(VAL):
        val_full[i] = R.run(p, SEED_A)[0].float()
    def pick(name, stats):
        best = None
        for alpha in a.alphas:
            W = solve(stats, alpha)
            errs = [float((R.run(p, SEED_A, blocks=blocks, k=k, mode='ridge', W=W)[0].float() - val_full[i]).square().mean()) for i, p in enumerate(VAL)]
            m = sum(errs) / len(errs)
            log(L, dict(stage='select', arm=name, alpha=alpha, val_latent_mse=m))
            if best is None or m < best[0]:
                best = (m, alpha, W)
        arms[name] = best
        torch.save({kk: v.half().cpu() for kk, v in best[2].items()}, out / f'W_{name}.pt')
        return best[2]
    W_off = pick('ridge_off', merge(sA))
    pick('ridge_off2x', merge(sA, sB))
    sOn1 = collect(SEED_B, False, W_off)
    W_dag = pick('ridge_dagger', merge(sA, sOn1))
    sOn2 = collect(SEED_A, False, W_dag)
    pick('ridge_dagger2', merge(sA, sOn1, sOn2))
    for name in ['zero', 'linear']:
        errs = [float((R.run(p, SEED_A, blocks=blocks, k=k, mode=name)[0].float() - val_full[i]).square().mean()) for i, p in enumerate(VAL)]
        log(L, dict(stage='select', arm=name, alpha=None, val_latent_mse=sum(errs) / len(errs)))
    json.dump({n: dict(val_latent_mse=b[0], alpha=b[1]) for n, b in arms.items()}, open(out / 'fit_summary.json', 'w'), indent=2)

def cmd_eval(a):
    out = ROOT / 'results/eval'; out.mkdir(parents=True, exist_ok=True)
    lat = ROOT / 'latents/eval'; lat.mkdir(parents=True, exist_ok=True)
    R = Runner(out); L = out / f'eval_shard{a.shard}.jsonl'
    done = set()
    for f in out.glob('eval_shard*.jsonl'):
        for line in open(f):
            r = json.loads(line); done.add((r['prompt'], r['seed'], r['method']))
    methods = [('full', {}, STEPS), ('full_perturb1e-3', dict(perturb=1e-3), STEPS), ('full_perturb1e-2', dict(perturb=1e-2), STEPS)] if a.part in ('base', 'all') else []
    for b, (blocks, k) in BUDGETS.items():
        if a.part in ('base', 'all'):
            methods += [(f'steps{matched_steps(b)}', {}, matched_steps(b)), (f'{b}_zero', dict(blocks=blocks, k=k, mode='zero'), STEPS),
                        (f'{b}_linear', dict(blocks=blocks, k=k, mode='linear'), STEPS)]
        if a.part == 'boost' and b in ('B2', 'B3'):
            for gm in (1.1, 1.25):
                methods.append((f'{b}_hiboost{gm}', dict(blocks=blocks, k=k, mode='hiboost', W=gm), STEPS))
        if a.part == 'window' and b in ('B2', 'B3'):
            for name, win in [('early', (1, 23)), ('late', (7, 29))]:
                methods.append((f'{b}_zero_{name}', dict(blocks=blocks, k=k, mode='zero', win=win), STEPS))
        if a.part == 'freq' and b in ('B2', 'B3'):
            methods += [(f'{b}_damped0.5', dict(blocks=blocks, k=k, mode='damped', W=0.5), STEPS),
                        (f'{b}_freqhi0.15', dict(blocks=blocks, k=k, mode='freq_hi', W=0.15), STEPS),
                        (f'{b}_freqlo0.15', dict(blocks=blocks, k=k, mode='freq_lo', W=0.15), STEPS)]
        if a.part in ('ridge', 'all'):
            for arm in ['ridge_off', 'ridge_off2x', 'ridge_dagger', 'ridge_dagger2']:
                Wp = ROOT / f'results/fit_{b}/W_{arm}.pt'
                if Wp.exists():
                    W = {kk: v.float().cuda() for kk, v in torch.load(Wp, weights_only=True).items()}
                    methods.append((f'{b}_{arm}', dict(blocks=blocks, k=k, mode='ridge', W=W), STEPS))
    seen = set(); methods = [m for m in methods if not (m[0] in seen or seen.add(m[0]))]
    jobs = [(i, s) for i in range(len(TEST)) for s in TEST_SEEDS]
    jobs = jobs[a.shard::a.nshards]
    for i, s in jobs:
        d = lat / f'p{i:02d}_s{s}'; d.mkdir(exist_ok=True)
        fullp = d / 'full.pt'
        for name, cfg, steps in methods:
            if (i, s, name) in done:
                continue
            z, info = R.run(TEST[i], s, steps=steps, **cfg)
            torch.save(z.cpu(), d / f'{name}.pt')
            mse = float((z.float().cpu() - torch.load(fullp, weights_only=True).float()).square().mean()) if name != 'full' else 0.
            log(L, dict(prompt=i, seed=s, method=name, steps=steps, latent_mse_to_full=mse, gpu=torch.cuda.get_device_name(), **info))

def cmd_score(a):
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    out = ROOT / 'results/score'; out.mkdir(parents=True, exist_ok=True)
    lat = ROOT / 'latents/eval'; gif = out / 'gifs'; gif.mkdir(exist_ok=True)
    L = out / f'score_shard{a.shard}.jsonl'
    done = set()
    for f in out.glob('score_shard*.jsonl'):
        for line in open(f):
            r = json.loads(line); done.add((r['prompt'], r['seed'], r['method']))
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL / 'vae', torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    clip = CLIPModel.from_pretrained(ROOT / 'weights/clip-vit-large-patch14', torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(ROOT / 'weights/clip-vit-large-patch14')
    from transformers import AutoModel, AutoImageProcessor
    dino = AutoModel.from_pretrained(ROOT / 'weights/dinov2-base', torch_dtype=torch.float16).cuda().eval()
    dproc = AutoImageProcessor.from_pretrained(ROOT / 'weights/dinov2-base')
    dirs = sorted(lat.glob('p*_s*'))[a.shard::a.nshards]
    with torch.no_grad():
        for d in dirs:
            i, s = int(d.name[1:3]), int(d.name.split('_s')[1])
            tf = clip.get_text_features(**{k: v.cuda() for k, v in proc(text=[TEST[i]], return_tensors='pt', padding=True, truncation=True).items()})
            tf = tf / tf.norm(dim=-1, keepdim=True)
            ref = None
            names = ['full'] + sorted(p.stem for p in d.glob('*.pt') if p.stem != 'full')
            for name in names:
                if (i, s, name) in done and name != 'full':
                    continue
                z = torch.load(d / f'{name}.pt', weights_only=True).float().cuda()
                v = vae.decode(z.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor).sample
                v = ((v.clamp(-1, 1) + 1) / 2)[0].cpu()  # C,T,H,W
                if name == 'full':
                    ref = v
                    if (i, s, name) in done:
                        continue
                frames = [Image.fromarray((v[:, t].permute(1, 2, 0) * 255).round().byte().numpy()) for t in range(v.shape[1])]
                pix = proc(images=frames[::4], return_tensors='pt')['pixel_values'].half().cuda()
                imf = clip.get_image_features(pixel_values=pix); imf = imf / imf.norm(dim=-1, keepdim=True)
                clip_score = float((imf @ tf.T).mean() * 100)
                cf = clip.get_image_features(pixel_values=proc(images=frames, return_tensors='pt')['pixel_values'].half().cuda())
                cf = cf / cf.norm(dim=-1, keepdim=True)
                df = dino(pixel_values=dproc(images=frames, return_tensors='pt')['pixel_values'].half().cuda()).last_hidden_state[:, 0]
                df = df / df.norm(dim=-1, keepdim=True)
                subject = float(((df[1:] * df[:-1]).sum(-1) + (df[1:] * df[:1]).sum(-1)).mean() / 2)
                background = float(((cf[1:] * cf[:-1]).sum(-1) + (cf[1:] * cf[:1]).sum(-1)).mean() / 2)
                gray = v.mean(0)
                motion = float((gray[1:] - gray[:-1]).abs().mean())
                lap = gray[:, 1:-1, 1:-1] * 4 - gray[:, :-2, 1:-1] - gray[:, 2:, 1:-1] - gray[:, 1:-1, :-2] - gray[:, 1:-1, 2:]
                sharpness = float(lap.var(dim=(1, 2)).mean())
                mse = float((v - ref).square().mean())
                tde = float(((v[:, 1:] - v[:, :-1]) - (ref[:, 1:] - ref[:, :-1])).square().mean())
                if i < 4 and s == TEST_SEEDS[0]:
                    small = [f.resize((360, 240)) for f in frames]
                    small[0].save(gif / f'p{i:02d}_{name}.gif', save_all=True, append_images=small[1:], duration=125, loop=0)
                log(L, dict(prompt=i, seed=s, method=name, clip_score=clip_score, subject_consistency=subject,
                            background_consistency=background, motion_magnitude=motion, sharpness=sharpness, rgb_mse_to_full=mse,
                            psnr_to_full=None if mse == 0 else float(-10 * torch.log10(torch.tensor(mse))), temporal_delta_error=tde))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest='cmd', required=True)
    sp.add_parser('embed')
    f = sp.add_parser('fit'); f.add_argument('--budget', required=True); f.add_argument('--alphas', type=float, nargs='+', default=[1e-3, 1e-2, 1e-1])
    e = sp.add_parser('eval'); e.add_argument('--shard', type=int, default=0); e.add_argument('--nshards', type=int, default=1); e.add_argument('--part', default='all')
    s = sp.add_parser('score'); s.add_argument('--shard', type=int, default=0); s.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    meta = ROOT / 'results/run_meta.jsonl'; meta.parent.mkdir(parents=True, exist_ok=True)
    with open(meta, 'a') as fh:
        fh.write(json.dumps(dict(cmd=a.cmd, args=vars(a), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), start=time.strftime('%F %T'))) + '\n')
    dict(embed=cmd_embed, fit=cmd_fit, eval=cmd_eval, score=cmd_score)[a.cmd](a)
