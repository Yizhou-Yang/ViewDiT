#!/usr/bin/env python3
"""ViewDiT H20 benchmark: our late-concentrated multi-granularity stack vs training-free cache baselines,
same model / same frames / same steps / same seeds / same GPU (paired wall time).

Models: CogVideoX-2b (fp16) / CogVideoX-5b (bf16).  env COG_MODEL=2b|5b, COG_F (frames, default 49),
COG_STEPS (default 50), COG_OUT, COG_LAT, COG_SUF (per-GPU shard suffix for jsonl files).

Baselines implemented inside the same executor (all training-free, re-implemented, not official code):
  steps{n}      : fewer DDIM steps
  TEA_{th}      : TeaCache (CogVideoX official polynomial on rel-L1 of timestep embedding; whole-stack residual reuse;
                  first/last step always computed).  Whole-stack residual == sum of all block residuals (exact).
  PAB_k{k}      : Pyramid-Attention-Broadcast style: reuse attention outputs (recompute FFN) every k steps for t in (100,800)
  FORA_n{n}     : static all-block residual reuse, refresh every n steps (FORA / Delta-DiT style)
  TS_n{n}       : TaylorSeer style: first-order Taylor forecast of all block residuals, refresh every n steps
  FC_*          : FasterCache style: CFG-uncond reuse (C) + attention reuse
Ours (late-concentrated):  O (whole-step skip, x0-space Taylor-1) + B (block residual reuse on remaining late steps)
  + optional G (guidance off late) -- only after warm step w; refresh rule: cond-only steps never refresh block cache.
"""
import argparse, glob, hashlib, json, math, os, time, types
from pathlib import Path
import torch

ROOT = Path('/data/workspace/viewdit')
MNAME = os.environ.get('COG_MODEL', '2b')
MODEL = ROOT / f'weights/CogVideoX-{MNAME}'
DT = torch.float16 if MNAME == '2b' else torch.bfloat16
F = int(os.environ.get('COG_F', '49'))
N = int(os.environ.get('COG_STEPS', '50'))
OUT = Path(os.environ.get('COG_OUT', str(ROOT / f'results/h20_bench/{MNAME}_f{F}_s{N}')))
LAT = Path(os.environ.get('COG_LAT', str(ROOT / f'latents/h20_bench/{MNAME}_f{F}_s{N}')))
SUF = os.environ.get('COG_SUF', '')
H, W_, CFG = 480, 720, 6.0
LT = (F - 1) // 4 + 1
TEA_COEF = {'2b': [-31.0658903, 25.4732368, -5.92380459, 1.75769064, -0.00361568434],
            '5b': [-1538.80483, 843.202495, -134.363087, 7.97131516, -0.0523162339]}
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


def block_fwd(blk, h, e, temb, rope, ah=None, ae=None, ff=None):
    """Exact copy of CogVideoXBlock.forward (diffusers 0.31) with optional cached operator outputs."""
    T_ = e.size(1)
    nh, ne_, g_msa, eg_msa = blk.norm1(h, e, temb)
    if ah is None:
        ah, ae = blk.attn1(hidden_states=nh, encoder_hidden_states=ne_, image_rotary_emb=rope)
    h2 = h + g_msa * ah
    e2 = e + eg_msa * ae
    nh2, ne2, g_ff, eg_ff = blk.norm2(h2, e2, temb)
    if ff is None:
        ff = blk.ff(torch.cat([ne2, nh2], dim=1))
    return h2 + g_ff * ff[:, T_:], e2 + eg_ff * ff[:, :T_], (ah, ae, ff)


class Executor:
    def __init__(self, dit):
        self.dit = dit
        self.blocks = dit.transformer_blocks
        self.reset()
        for i, b in enumerate(self.blocks):
            def fwd(block, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, idx=i):
                return self._fwd(idx, hidden_states, encoder_hidden_states, temb, image_rotary_emb)
            b.forward = types.MethodType(fwd, b)

    def reset(self, bmode='zero', damp=0.5, keep_ops=False):
        self.mem, self.bmode, self.damp, self.keep_ops = {}, bmode, damp, keep_ops
        self.step, self.branches, self.reuse_now = -1, ['u', 'c'], set()
        self.calls = dict(full=0, reuse=0, attn_only=0, ffn_only=0)

    def _fwd(self, idx, h, e, temb, rope):
        br = self.branches
        ready = all((idx, b) in self.mem for b in br)
        if idx not in self.reuse_now or not ready:
            ho, eo, (ah, ae, ff) = block_fwd(self.blocks[idx], h, e, temb, rope)
            self.calls['full'] += len(br)
            for k, b in enumerate(br):
                old = self.mem.get((idx, b))
                m = dict(dh=(ho - h)[k:k + 1].detach(), de=(eo - e)[k:k + 1].detach(), s=self.step,
                         pdh=None if old is None else old['dh'], pde=None if old is None else old['de'],
                         ps=None if old is None else old['s'])
                if self.keep_ops:
                    m.update(ah=ah[k:k + 1].detach(), ae=ae[k:k + 1].detach(), ff=ff[k:k + 1].detach())
                if self.bmode != 't1':
                    m['pdh'] = m['pde'] = None  # save memory
                self.mem[(idx, b)] = m
            return ho, eo
        ms = [self.mem[(idx, b)] for b in br]
        cat = lambda key: torch.cat([m[key] for m in ms])
        mode = self.bmode
        if mode == 'attn':
            self.calls['attn_only'] += len(br)
            ho, eo, _ = block_fwd(self.blocks[idx], h, e, temb, rope, ah=cat('ah'), ae=cat('ae'))
            return ho, eo
        self.calls['reuse'] += len(br)
        dh, de = cat('dh'), cat('de')
        if mode == 't1' and all(m['pdh'] is not None for m in ms):
            r = torch.tensor([(self.step - m['s']) / (m['s'] - m['ps']) for m in ms], device=h.device, dtype=h.dtype)[:, None, None]
            dh = dh + self.damp * r * (dh - cat('pdh'))
            de = de + self.damp * r * (de - cat('pde'))
        return h + dh, e + de


def rope_for(dit, device):
    if not dit.config.use_rotary_positional_embeddings:
        return None
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
    p = dit.config.patch_size
    gh, gw = H // (8 * p), W_ // (8 * p)
    crop = get_resize_crop_region_for_grid((gh, gw), 720 // (8 * p), 480 // (8 * p))
    c, s = get_3d_rotary_pos_embed(embed_dim=dit.config.attention_head_dim, crops_coords=crop, grid_size=(gh, gw), temporal_size=LT)
    return c.to(device), s.to(device)


def load_dit(cls):
    d = MODEL / 'transformer'
    if (d / 'diffusion_pytorch_model.safetensors').exists():
        return cls.from_pretrained(d, torch_dtype=DT).cuda().eval().requires_grad_(False)
    from safetensors.torch import load_file
    m = cls.from_config(cls.load_config(d)).to(DT)
    sd = {}
    for fn in sorted(glob.glob(str(d / 'diffusion_pytorch_model-*.safetensors'))):
        sd.update(load_file(fn))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and not [k for k in missing if 'pos_embedding' not in k], (missing[:5], unexpected[:5])
    return m.cuda().eval().requires_grad_(False)


class Sampler:
    def __init__(self):
        from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
        torch.set_num_threads(8)
        self.dit = load_dit(CogVideoXTransformer3DModel)
        self.sch = CogVideoXDDIMScheduler.from_pretrained(MODEL / 'scheduler')
        self.NL = len(self.dit.transformer_blocks)
        self.ex = Executor(self.dit)
        self.emb = torch.load(OUT / 'embeddings.pt', map_location='cpu', weights_only=True)
        self.rope = rope_for(self.dit, 'cuda')

    def dit_call(self, lat, pe_list, t):
        x = torch.cat([lat] * len(pe_list))
        e = torch.cat(pe_list)
        return self.dit(hidden_states=x, encoder_hidden_states=e, timestep=t.expand(x.shape[0]), image_rotary_emb=self.rope,
                        return_dict=False)[0].float()

    @torch.no_grad()
    def tea_schedule(self, steps, th, w=0):
        """TeaCache decisions depend only on the timestep embedding -> deterministic per schedule.
        w>0: fairness variant that forbids skipping before warm step w (same late-concentration prior as ours)."""
        self.sch.set_timesteps(steps, device='cuda')
        c = TEA_COEF[MNAME]
        poly = lambda x: c[0] * x ** 4 + c[1] * x ** 3 + c[2] * x ** 2 + c[3] * x + c[4]
        prev, acc, skip = None, 0., []
        for i, t in enumerate(self.sch.timesteps):
            emb = self.dit.time_embedding(self.dit.time_proj(t[None]).to(DT), None).float()
            if i == 0 or i == steps - 1 or i < w:
                acc = 0.
            else:
                acc += poly(float((emb - prev).abs().mean() / prev.abs().mean()))
                if acc < th:
                    skip.append(i)
                else:
                    acc = 0.
            prev = emb
        return skip

    @torch.no_grad()
    def get_sch(self, solver):
        """Higher-order solvers on the SAME noise schedule (CogVideoX snr-shift + zero-SNR alphas via trained_betas)."""
        if not solver:
            return self.sch
        self._sch = getattr(self, '_sch', {})
        if solver not in self._sch:
            from diffusers import DPMSolverMultistepScheduler, UniPCMultistepScheduler
            ac = self.sch.alphas_cumprod.clone().double()
            ac[-1] = max(float(ac[-1]), 2 ** -24)
            prev = torch.cat([torch.ones(1, dtype=ac.dtype), ac[:-1]])
            kw = dict(num_train_timesteps=len(ac), trained_betas=(1 - ac / prev).float().tolist(), prediction_type='v_prediction',
                      timestep_spacing='trailing', final_sigmas_type='zero')
            if solver == 'dpm':
                self._sch[solver] = DPMSolverMultistepScheduler(algorithm_type='dpmsolver++', solver_order=2, **kw)
            elif solver == 'dpm3':
                self._sch[solver] = DPMSolverMultistepScheduler(algorithm_type='dpmsolver++', solver_order=3, **kw)
            else:
                self._sch[solver] = UniPCMultistepScheduler(solver_order=2, **kw)
        return self._sch[solver]

    @torch.no_grad()
    def run(self, prompt, seed, plan):
        steps = plan.get('steps', N)
        solver = plan.get('solver')
        sch = self.get_sch(solver)
        self.sch.set_timesteps(steps, device='cuda')
        sch.set_timesteps(steps, device='cuda')
        ts = sch.timesteps
        self.ex.reset(plan.get('bmode', 'zero'), plan.get('damp', 0.5), plan.get('bmode') == 'attn')
        pe, ne = self.emb[prompt].cuda().to(DT), self.emb[''].cuda().to(DT)
        g = torch.Generator(device='cuda').manual_seed(seed)
        lat = torch.randn((1, LT, 16, 60, 90), generator=g, device='cuda', dtype=DT)
        if plan.get('perturb'):
            eps = torch.randn(lat.shape, generator=torch.Generator(device='cuda').manual_seed(seed + 7), device='cuda', dtype=torch.float32)
            lat = ((lat.float() + plan['perturb'] * eps) / (1 + plan['perturb'] ** 2) ** .5).to(DT)
        lat = lat * self.sch.init_noise_sigma
        C = set(plan.get('C', ())); G = set(plan.get('G', ())); O = set(plan.get('O', ()))
        breuse = {int(k): set(v) for k, v in plan.get('B', {}).items()}
        ford, odamp = plan.get('ford', 'x01'), plan.get('odamp', 1.)
        hist, delta = [], None
        kinds = dict(F=0, C=0, G=0, O=0)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i, t in enumerate(ts):
            ab = float(self.sch.alphas_cumprod[int(t)])
            x = lat.float()
            if i in O and len(hist) >= 1:
                kinds['O'] += 1
                j, vj, x0j = hist[-1]
                if ford == 'reuse' or len(hist) < 2:
                    v = vj
                else:
                    j1, vj1, x0j1 = hist[-2]
                    r = odamp * (i - j) / (j - j1)
                    if ford == 'v1':
                        v = vj + r * (vj - vj1)
                    else:
                        x0 = x0j + r * (x0j - x0j1)
                        v = (ab ** .5 * x - x0) / (1 - ab) ** .5
            else:
                self.ex.step = i
                self.ex.reuse_now = breuse.get(i, set())
                if i in G:
                    kinds['G'] += 1
                    self.ex.branches = ['c']
                    v = self.dit_call(lat, [pe], t)
                elif i in C and delta is not None:
                    kinds['C'] += 1
                    self.ex.branches = ['c']
                    c = self.dit_call(lat, [pe], t)
                    u = c + delta
                    v = u + CFG * (c - u)
                else:
                    kinds['F'] += 1
                    self.ex.branches = ['u', 'c']
                    u, c = self.dit_call(lat, [ne, pe], t).chunk(2)
                    delta = u - c
                    v = u + CFG * (c - u)
                hist = (hist + [(i, v, ab ** .5 * x - (1 - ab) ** .5 * v)])[-2:]
            if solver:
                lat = sch.step(v, t, lat.float(), return_dict=False)[0].to(DT)
            else:
                lat = self.sch.step(v, t, lat, eta=0., return_dict=False)[0].to(DT)
        torch.cuda.synchronize()
        assert torch.isfinite(lat.float()).all()
        return lat, dict(seconds=time.perf_counter() - t0, kinds=kinds, **self.ex.calls)


def sc(s):
    """scale a 30-step index to N steps."""
    return int(round(s * N / 30))


def lateplan(w, kr=4, ko=2, gi=None, od=None, bmode=None, blocks=None, NL=30):
    blocks = set(range(NL)) if blocks is None else blocks
    O = [s for s in range(w + 1, N) if (s - w) % ko]
    real = [s for s in range(w, N) if s not in O]
    per = (kr // 2 if ko == 2 else kr) if kr else 0
    B = {s: blocks for n, s in enumerate(real) if per and n % per != 0}
    pl = dict(O=O, ford='x01', B=B)
    if bmode: pl['bmode'] = bmode
    if gi is not None: pl['G'] = [s for s in real if s >= gi]
    if od: pl['odamp'] = od
    return pl


def PLANS(phase, S=None):
    NL = S.NL if S else 30
    allb = set(range(NL))
    P = {'full': {}}
    tea = lambda th: dict(B={s: allb for s in S.tea_schedule(N, th)}) if S else {}
    if phase in ('fair', 'fairbench'):
        P['full_perturb1e-2'] = dict(perturb=1e-2)
    if phase == 'fair':
        # (1) fairness: competitors get the same late-concentration prior (warm w) and late guidance-off, with a wide knob sweep
        for w30 in (6, 10):
            w = sc(w30)
            for th in (0.1, 0.2, 0.3, 0.5, 0.8):
                P[f'TEAw{w}_{th}'] = dict(B={s: allb for s in S.tea_schedule(N, th, w)}) if S else {}
            for th in (0.2, 0.5):
                pl = dict(B={s: allb for s in S.tea_schedule(N, th, w)}) if S else {}
                pl['G'] = list(range(sc(22), N))
                P[f'TEAw{w}_{th}_GI'] = pl
            for kr in (2, 3, 4):
                P[f'FORAw{w}_n{kr}'] = lateplan(w, kr=kr, ko=1, NL=NL)          # ablation: B only (late static reuse)
            for kr in (2, 3):
                P[f'TSw{w}_n{kr}'] = lateplan(w, kr=kr, ko=1, bmode='t1', NL=NL)  # TaylorSeer + warm
            for ko in (2, 3):
                P[f'Oonly_O{ko}w{w}'] = lateplan(w, kr=0, ko=ko, NL=NL)          # ablation: O only (x0 Taylor skip)
        for th in (0.4, 0.5, 0.8):
            P[f'TEA_{th}'] = tea(th)
        # (2) higher-order ODE solvers on the same schedule
        for s in (15, 20, 25, 30):
            P[f'dpm{s}'] = dict(steps=s, solver='dpm')
            P[f'unipc{s}'] = dict(steps=s, solver='unipc')
        P['dpm3_20'] = dict(steps=20, solver='dpm3')
        # anchors (ours, already selected)
        for k in ('OURS_Ow20_L4', 'OURS_Ow17_L4_GI', 'OURS_O3w13_B2', 'OURS_O4w13_B2_GI', 'OURS_O3w10_B3'):
            P[k] = {**PLANS('tune', S), **PLANS('tune2', S)}[k]
        # (3) our stack on top of a higher-order solver is not defined (O uses DDIM x0 extrapolation) -> not tested
    if phase == 'fairbench':
        fp = OUT.parent / f'{MNAME}_f{F}_s{N}_fairsel.json'
        base = {**PLANS('fair', S), **PLANS('tune', S), **PLANS('tune2', S)}
        for k in json.load(open(fp)):
            P[k] = base[k]
    if phase in ('tune', 'bench', 'bench5b'):
        P['full_perturb1e-2'] = dict(perturb=1e-2)
    if phase == 'tune':
        # re-tune our late plans at N steps + calibrate baselines' knobs to speed tiers (prompts 0-3)
        for th in (0.05, 0.1, 0.15, 0.2, 0.3):
            P[f'TEA_{th}'] = tea(th)
        for w30 in (6, 8, 10, 12):
            w = sc(w30)
            P[f'OURS_Ow{w}_L4'] = lateplan(w, NL=NL)
            P[f'OURS_O3w{w}_B2'] = lateplan(w, kr=2, ko=3, NL=NL)
            P[f'OURS_O3w{w}_B3'] = lateplan(w, kr=3, ko=3, NL=NL)
        P[f'OURS_Ow{sc(10)}_L4_GI'] = lateplan(sc(10), gi=sc(22), NL=NL)
        P[f'OURS_O3w{sc(10)}_B2_GI'] = lateplan(sc(10), kr=2, ko=3, gi=sc(22), NL=NL)
        P[f'OURS_O4w{sc(10)}_B2'] = lateplan(sc(10), kr=2, ko=4, NL=NL)
        P[f'OURS_O4w{sc(8)}_B2'] = lateplan(sc(8), kr=2, ko=4, NL=NL)
        P[f'OURS_Ow{sc(10)}_L6'] = lateplan(sc(10), kr=6, NL=NL)
        P[f'OURS_Ow{sc(4)}_L6_GI'] = lateplan(sc(4), kr=6, gi=sc(22), NL=NL)
    if phase == 'tune2':
        for w30 in (6, 8):
            P[f'OURS_O4w{sc(w30)}_B2'] = lateplan(sc(w30), kr=2, ko=4, NL=NL)
            P[f'OURS_O4w{sc(w30)}_B3'] = lateplan(sc(w30), kr=3, ko=4, NL=NL)
        P[f'OURS_O5w{sc(8)}_B2'] = lateplan(sc(8), kr=2, ko=5, NL=NL)
        P[f'OURS_O4w{sc(8)}_B2_GI'] = lateplan(sc(8), kr=2, ko=4, gi=sc(22), NL=NL)
        P[f'OURS_O4w{sc(10)}_B3'] = lateplan(sc(10), kr=3, ko=4, NL=NL)
    if phase in ('tune2', 'bench', 'bench5b'):
        tmid = [i for i in range(N) if 100 < 1000 * (1 - i / N) < 800]
        # FasterCache-style, refresh-rule respected: attn refresh steps full (u,c); reuse steps = cond-only + uncond from CFG delta
        P['FC_v2'] = dict(B={s: allb for n, s in enumerate(tmid) if n % 2}, bmode='attn', C=[s for n, s in enumerate(tmid) if n % 2])
        P['FC_v2_k3'] = dict(B={s: allb for n, s in enumerate(tmid) if n % 3}, bmode='attn', C=[s for n, s in enumerate(tmid) if n % 3])
    if phase in ('tune', 'bench', 'bench5b'):
        for s in (N // 2, int(N * 0.4), int(N * 0.34)):
            P[f'steps{s}'] = dict(steps=s)
        tmid = [i for i in range(N) if 100 < 1000 * (1 - i / N) < 800]  # trailing timesteps ~ 1000*(1-i/N)
        for k in (2, 3):
            P[f'PAB_k{k}'] = dict(B={s: allb for n, s in enumerate(tmid) if n % k}, bmode='attn')
        for n in (2, 3):
            P[f'FORA_n{n}'] = dict(B={s: allb for s in range(2, N - 1) if (s - 2) % n})
        for n in (3, 4):
            P[f'TS_n{n}'] = dict(B={s: allb for s in range(4, N - 1) if (s - 4) % n}, bmode='t1', damp=1.0)
        # FasterCache style: uncond reuse on odd steps after 1/3 + attention reuse on same steps' neighbours (refresh rule kept)
        P['FC_cfg'] = dict(C=[s for s in range(N // 3, N) if s % 2])
        P['FC_cfg+pab'] = dict(C=[s for s in range(N // 3, N) if s % 2],
                               B={s: allb for s in tmid if s % 2 == 0 and s > tmid[0]}, bmode='attn')
    if phase in ('bench', 'bench5b'):
        for th in (0.1, 0.15, 0.2, 0.3):
            P[f'TEA_{th}'] = tea(th)
        sel = json.load(open(OUT.parent / f'{MNAME}_f{F}_s{N}_selected.json')) if (OUT.parent / f'{MNAME}_f{F}_s{N}_selected.json').exists() else None
        base = {**PLANS('tune', S), **PLANS('tune2', S)}
        keys = sel or [k for k in base if k.startswith('OURS_')]
        for k in keys:
            if k in base:
                P[k] = base[k]
    return P


def jsonl(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def done_set(pat):
    s = set()
    for fn in glob.glob(str(pat)):
        for line in open(fn):
            r = json.loads(line); s.add((r['prompt'], r['seed'], r['method']))
    return s


def cmd_embed(a):
    from transformers import T5EncoderModel, T5Tokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(MODEL / 'tokenizer')
    enc = T5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=DT).cuda().eval()
    emb = {}
    with torch.no_grad():
        for p in ['', *TEST]:
            ids = tok(p, padding='max_length', max_length=226, truncation=True, return_tensors='pt').input_ids.cuda()
            emb[p] = enc(ids)[0].to(DT).cpu()
    torch.save(emb, OUT / 'embeddings.pt')
    print('embeddings', len(emb))


def cmd_verify(a):
    from diffusers import CogVideoXPipeline
    S = Sampler()
    p = TEST[0]
    z, info = S.run(p, 3000, {})
    class DP(CogVideoXPipeline):
        @property
        def _execution_device(self):
            return torch.device('cuda')
    S.ex.reset()
    pipe = DP(vae=None, transformer=S.dit, scheduler=S.sch, tokenizer=None, text_encoder=None)
    pipe.vae_scale_factor_spatial, pipe.vae_scale_factor_temporal = 8, 4
    pipe.set_progress_bar_config(disable=True)
    init = torch.randn((1, LT, 16, 60, 90), generator=torch.Generator(device='cuda').manual_seed(3000), device='cuda', dtype=DT)
    zp = pipe(latents=init, prompt_embeds=S.emb[p].cuda().to(DT), negative_prompt_embeds=S.emb[''].cuda().to(DT), height=H, width=W_, num_frames=F,
              num_inference_steps=N, guidance_scale=CFG, eta=0., output_type='latent', generator=torch.Generator(device='cuda').manual_seed(3000)).frames
    d = float((z.float() - zp.float()).abs().max()); rel = float((z.float() - zp.float()).abs().mean() / zp.float().abs().mean())
    r = dict(model=MNAME, F=F, N=N, max_abs_diff_vs_pipeline=d, rel_mean_diff=rel, seconds_full=info['seconds'], gpu=torch.cuda.get_device_name(), NL=S.NL)
    (OUT / 'verify.json').write_text(json.dumps(r, indent=1))
    print(r)


def dump_plan(v):
    o = {}
    for kk, vv in v.items():
        if isinstance(vv, (set, list)):
            o[kk] = sorted(vv)
        elif isinstance(vv, dict):
            o[kk] = {str(s): (sorted(b) if isinstance(b, (set, list)) else b) for s, b in vv.items()}
        else:
            o[kk] = vv
    return o


def cmd_gen(a):
    OUT.mkdir(parents=True, exist_ok=True)
    S = Sampler()
    P = PLANS(a.phase, S)
    if a.only:
        P = {k: v for k, v in P.items() if k in a.only.split(',') or k == 'full'}
    L = OUT / f'gen_{a.phase}{SUF}.jsonl'
    done = done_set(OUT / f'gen_{a.phase}*.jsonl')
    json.dump({k: dump_plan(v) for k, v in P.items()}, open(OUT / f'plans_{a.phase}.json', 'w'), indent=0)
    prompts = a.plist if a.plist else list(range(a.p0, a.p0 + a.nprompt))
    for i in prompts:
        for seed in a.seeds:
            d = LAT / f'p{i:02d}_s{seed}'; d.mkdir(parents=True, exist_ok=True)
            for name, plan in P.items():
                if (i, seed, name) in done:
                    continue
                z, info = S.run(TEST[i], seed, plan)
                torch.save(z.cpu(), d / f'{name}.pt')
                mse = 0. if name == 'full' else float((z.float().cpu() - torch.load(d / 'full.pt', weights_only=True).float()).square().mean())
                jsonl(L, dict(phase=a.phase, prompt=i, seed=seed, method=name, latent_mse_to_full=mse, **info))


def ssim_video(x, y):
    """x,y: [T,3,H,W] in [0,1] on cuda; gaussian 11x11 sigma 1.5, per channel, mean."""
    import torch.nn.functional as Fn
    k = torch.arange(11, device=x.device, dtype=torch.float32) - 5
    g = torch.exp(-k ** 2 / (2 * 1.5 ** 2)); g = g / g.sum()
    win = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda z: Fn.conv2d(z, win, groups=3)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return float(s.mean())


def cmd_score(a):
    from diffusers import AutoencoderKLCogVideoX
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    import lpips
    L = OUT / f'score_{a.phase}{SUF}.jsonl'
    done = done_set(OUT / f'score_{a.phase}*.jsonl')
    gen = [json.loads(l) for l in open(OUT / f'gen_{a.phase}{SUF}.jsonl')]
    todo = sorted({(r['prompt'], r['seed']) for r in gen})
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL / 'vae', torch_dtype=DT).cuda().eval().requires_grad_(False)
    vae.enable_tiling()
    cp = ROOT / 'weights/clip-vit-large-patch14'
    clip = CLIPModel.from_pretrained(cp, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(cp)
    lp = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    gif = OUT / 'gifs'; gif.mkdir(exist_ok=True)
    with torch.no_grad():
        for i, s in todo:
            d = LAT / f'p{i:02d}_s{s}'
            names = ['full'] + sorted({r['method'] for r in gen if r['prompt'] == i and r['seed'] == s} - {'full'})
            if all((i, s, n) in done for n in names):
                continue
            tf = clip.get_text_features(**{k: v.cuda() for k, v in proc(text=[TEST[i]], return_tensors='pt', padding=True, truncation=True).items()})
            tf = tf / tf.norm(dim=-1, keepdim=True)
            ref = None
            for name in names:
                if (i, s, name) in done and name != 'full':
                    continue
                z = torch.load(d / f'{name}.pt', weights_only=True).to(DT).cuda()
                v = vae.decode(z.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor).sample.float()
                v = ((v.clamp(-1, 1) + 1) / 2)[0]  # [3,T,H,W] cuda
                if name == 'full':
                    ref = v
                    if (i, s, name) in done:
                        continue
                vc = v.cpu()
                frames = [Image.fromarray((vc[:, t].permute(1, 2, 0) * 255).round().byte().numpy()) for t in range(vc.shape[1])]
                imf = clip.get_image_features(pixel_values=proc(images=frames, return_tensors='pt')['pixel_values'].half().cuda())
                imf = imf / imf.norm(dim=-1, keepdim=True)
                gray = v.mean(0)
                lap = gray[:, 1:-1, 1:-1] * 4 - gray[:, :-2, 1:-1] - gray[:, 2:, 1:-1] - gray[:, 1:-1, :-2] - gray[:, 1:-1, 2:]
                mse = float((v - ref).square().mean())
                vt, rt = v.permute(1, 0, 2, 3), ref.permute(1, 0, 2, 3)
                lpv = float(torch.cat([lp(vt[k:k + 8] * 2 - 1, rt[k:k + 8] * 2 - 1).flatten() for k in range(0, vt.shape[0], 8)]).mean())
                if i < 2 and s == min(r['seed'] for r in gen):
                    small = [f.resize((360, 240)) for f in frames[::2]]
                    small[0].save(gif / f'{a.phase}_p{i:02d}_{name}.gif', save_all=True, append_images=small[1:], duration=125, loop=0)
                jsonl(L, dict(phase=a.phase, prompt=i, seed=s, method=name, clip_score=float((imf @ tf.T).mean() * 100),
                              frame_consistency=float(((imf[1:] * imf[:-1]).sum(-1)).mean()),
                              motion=float((gray[1:] - gray[:-1]).abs().mean()), sharpness=float(lap.var(dim=(1, 2)).mean()),
                              psnr_to_full=None if mse == 0 else float(-10 * math.log10(mse)),
                              ssim_to_full=ssim_video(vt, rt), lpips_to_full=lpv,
                              temporal_delta_error=float(((v[:, 1:] - v[:, :-1]) - (ref[:, 1:] - ref[:, :-1])).square().mean())))


def cmd_summary(a):
    import numpy as np
    gen = [json.loads(l) for fn in glob.glob(str(OUT / f'gen_{a.phase}*.jsonl')) for l in open(fn)]
    sc_ = {}
    for fn in glob.glob(str(OUT / f'score_{a.phase}*.jsonl')):
        for r in map(json.loads, open(fn)):
            sc_[(r['prompt'], r['seed'], r['method'])] = r
    full_t = {(r['prompt'], r['seed']): r['seconds'] for r in gen if r['method'] == 'full'}
    fs = {k[:2]: v for k, v in sc_.items() if k[2] == 'full'}
    rows = {}
    for r in gen:
        k = (r['prompt'], r['seed'])
        if k not in full_t:
            continue
        e = rows.setdefault(r['method'], dict(speed=[], sec=[], psnr=[], ssim=[], lpips=[], dclip=[], sharp=[], motion=[], tde=[]))
        e['speed'].append(full_t[k] / r['seconds']); e['sec'].append(r['seconds'])
        s = sc_.get((*k, r['method']))
        if s and k in fs:
            if s['psnr_to_full'] is not None:
                e['psnr'].append(s['psnr_to_full']); e['ssim'].append(s['ssim_to_full']); e['lpips'].append(s['lpips_to_full'])
            e['dclip'].append(s['clip_score'] - fs[k]['clip_score'])
            e['sharp'].append(s['sharpness'] / fs[k]['sharpness'])
            e['motion'].append(s['motion'] / max(fs[k]['motion'], 1e-9))
            e['tde'].append(s['temporal_delta_error'])
    m = lambda x: float(np.mean(x)) if x else float('nan')
    lines = [f'# CogVideoX-{MNAME} bench ({a.phase}), {F}f 480x720, DDIM-{N}, CFG6, H20, paired wall-time speed (DiT only)', '',
             '| method | n | speed | sec | PSNR | SSIM | LPIPS | dCLIP | sharp | motion | tdelta err |', '|---|---|---|---|---|---|---|---|---|---|---|']
    res = {}
    for k, e in sorted(rows.items(), key=lambda kv: -m(kv[1]['speed'])):
        res[k] = {kk: m(v) for kk, v in e.items()}; res[k]['n'] = len(e['speed']); res[k]['n_scored'] = len(e['psnr'])
        lines.append(f"| {k} | {len(e['speed'])}/{len(e['psnr'])} | {m(e['speed']):.2f} | {m(e['sec']):.1f} | {m(e['psnr']):.2f} | {m(e['ssim']):.4f} | "
                     f"{m(e['lpips']):.4f} | {m(e['dclip']):+.2f} | {m(e['sharp']):.3f} | {m(e['motion']):.3f} | {m(e['tde']):.5f} |")
    (OUT / f'SUMMARY_{a.phase}.md').write_text('\n'.join(lines) + '\n')
    json.dump(res, open(OUT / f'summary_{a.phase}.json', 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest='cmd', required=True)
    sp.add_parser('embed'); sp.add_parser('verify')
    for n in ('gen', 'score', 'summary', 'plans'):
        q = sp.add_parser(n); q.add_argument('--phase', default='tune')
        q.add_argument('--nprompt', type=int, default=4); q.add_argument('--p0', type=int, default=0)
        q.add_argument('--plist', type=int, nargs='*', default=[]); q.add_argument('--seeds', type=int, nargs='+', default=[3000])
        q.add_argument('--only', default='')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'run_meta.jsonl', 'a') as fh:
        fh.write(json.dumps(dict(cmd=a.cmd, model=MNAME, F=F, N=N, suf=SUF, args=vars(a), sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 start=time.strftime('%F %T'))) + '\n')
    if a.cmd == 'plans':
        S = Sampler(); P = PLANS(a.phase, S)
        for k, v in P.items():
            nB = sum(len(b) for b in v.get('B', {}).values()) / S.NL
            print(k, 'O', len(v.get('O', [])), 'G', len(v.get('G', [])), 'C', len(v.get('C', [])), 'Bsteps', round(nB, 1), 'steps', v.get('steps', N))
    else:
        dict(embed=cmd_embed, verify=cmd_verify, gen=cmd_gen, score=cmd_score, summary=cmd_summary)[a.cmd](a)
