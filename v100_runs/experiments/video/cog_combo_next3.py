#!/usr/bin/env python3
"""CogVideoX-2b component-combination screen on V100 (DiT only; VAE used only for measurement).

Own DDIM sampling loop (verified equal to CogVideoXPipeline) so that components at four
granularities can be freely combined inside one run:
  step level   : O  output-level forecast (skip whole DiT call); reuse / Taylor-1 in v or x0 space;
                    optional low-weight retrospective correction (retro_w)
  branch level : C  CFG uncond reuse  u_t = c_t + (u - c)_last   (FasterCache-style)
                 G  guidance off (cond only, v = c)                 (guidance-interval-style)
  block level  : block residual reuse on scheduled (step, block): zero / damped Taylor-1
  operator lvl : attn-only reuse (recompute FFN) / ffn-only reuse (recompute attention)
Plans are dicts; see PLANS(). Components are not claimed novel individually.
"""
import argparse, hashlib, json, math, os, time, types
from pathlib import Path
import torch

ROOT = Path('/root/viewdit')
MODEL = ROOT / 'weights/CogVideoX-2b'
OUT = Path(os.environ.get('COG_OUT', str(ROOT / 'results/video/cog_combo_20260924')))
LAT = Path(os.environ.get('COG_LAT', str(ROOT / 'latents/cog_combo_20260924')))
H, W_, CFG = 480, 720, 6.0
F = int(os.environ.get('COG_F', '17'))
LT = (F - 1) // 4 + 1
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
        self.step, self.branches, self.reuse_now, self.reuse_u_now = -1, ['u', 'c'], set(), set()
        self.calls = dict(full=0, reuse=0, attn_only=0, ffn_only=0)

    def _fwd(self, idx, h, e, temb, rope):
        br = self.branches
        ready = all((idx, b) in self.mem for b in br)
        if br == ['u', 'c'] and idx in self.reuse_u_now and ready and idx not in self.reuse_now:
            # asymmetric: uncond branch reuses its cached residual, cond branch recomputed
            ho_c, eo_c, (ah, ae, ff) = block_fwd(self.blocks[idx], h[1:2], e[1:2], temb[1:2], rope)
            self.calls['full'] += 1; self.calls['reuse'] += 1
            old = self.mem.get((idx, 'c'))
            self.mem[(idx, 'c')] = dict(dh=(ho_c - h[1:2]).detach(), de=(eo_c - e[1:2]).detach(), s=self.step,
                                        pdh=old['dh'], pde=old['de'], ps=old['s'])
            mu = self.mem[(idx, 'u')]
            return torch.cat([h[0:1] + mu['dh'], ho_c]), torch.cat([e[0:1] + mu['de'], eo_c])
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
                self.mem[(idx, b)] = m
            return ho, eo
        ms = [self.mem[(idx, b)] for b in br]
        cat = lambda key: torch.cat([m[key] for m in ms])
        mode = self.bmode
        if mode == 'attn':
            self.calls['attn_only'] += len(br)
            ho, eo, _ = block_fwd(self.blocks[idx], h, e, temb, rope, ah=cat('ah'), ae=cat('ae'))
            return ho, eo
        if mode == 'ffn':
            self.calls['ffn_only'] += len(br)
            ho, eo, _ = block_fwd(self.blocks[idx], h, e, temb, rope, ff=cat('ff'))
            return ho, eo
        self.calls['reuse'] += len(br)
        dh, de = cat('dh'), cat('de')
        if mode == 't1' and all(m['pdh'] is not None for m in ms):
            r = torch.tensor([(self.step - m['s']) / (m['s'] - m['ps']) for m in ms], device=h.device, dtype=h.dtype)[:, None, None]
            dh = dh + self.damp * r * (dh - cat('pdh'))
            de = de + self.damp * r * (de - cat('pde'))
        return h + dh, e + de


class Sampler:
    def __init__(self):
        from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
        torch.set_num_threads(8)
        self.dit = CogVideoXTransformer3DModel.from_pretrained(MODEL / 'transformer', torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
        self.sch = CogVideoXDDIMScheduler.from_pretrained(MODEL / 'scheduler')
        self.ex = Executor(self.dit)
        self.emb = torch.load(OUT / 'embeddings.pt', map_location='cpu', weights_only=True)
        self.calib = {}

    def dit_call(self, lat, pe_list, t):
        x = torch.cat([lat] * len(pe_list))
        e = torch.cat(pe_list)
        return self.dit(hidden_states=x, encoder_hidden_states=e, timestep=t.expand(x.shape[0]), return_dict=False)[0].float()

    def coef(self, t):
        """x_next = A*x + Cv*v for CogVideoX DDIM (v-pred, eta=0)."""
        s = self.sch
        prev = t - s.config.num_train_timesteps // s.num_inference_steps
        ab = s.alphas_cumprod[t]
        abp = s.alphas_cumprod[prev] if prev >= 0 else s.final_alpha_cumprod
        a_t = ((1 - abp) / (1 - ab)) ** 0.5
        b_t = abp ** 0.5 - ab ** 0.5 * a_t
        return float(a_t + b_t * ab ** 0.5), float(-b_t * (1 - ab) ** 0.5), float(ab)

    @torch.no_grad()
    def run(self, prompt, seed, plan):
        steps = plan.get('steps', 30)
        self.sch.set_timesteps(steps, device='cuda')
        ts = self.sch.timesteps
        self.ex.reset(plan.get('bmode', 'zero'), plan.get('damp', 0.5), plan.get('bmode') in ('attn', 'ffn'))
        pe, ne = self.emb[prompt].cuda(), self.emb[''].cuda()
        g = torch.Generator(device='cuda').manual_seed(seed)
        lat = torch.randn((1, LT, 16, 60, 90), generator=g, device='cuda', dtype=torch.float16)
        if plan.get('perturb'):
            eps = torch.randn(lat.shape, generator=torch.Generator(device='cuda').manual_seed(seed + 7), device='cuda', dtype=torch.float32)
            lat = ((lat.float() + plan['perturb'] * eps) / (1 + plan['perturb'] ** 2) ** .5).half()
        lat = lat * self.sch.init_noise_sigma
        C = set(plan.get('C', ())); G = set(plan.get('G', ())); O = set(plan.get('O', ()))
        breuse = {int(k): set(v) for k, v in plan.get('B', {}).items()}
        bureuse = {int(k): set(v) for k, v in plan.get('BU', {}).items()}
        ford, retro_w = plan.get('ford', 'v1'), plan.get('retro_w', 0.)
        odamp, cmode, cdamp = plan.get('odamp', 1.), plan.get('cmode', 'zero'), plan.get('cdamp', 0.5)
        hist, delta, pend, A_log = [], None, [], {}
        dhist = []
        kinds = dict(F=0, C=0, G=0, O=0)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i, t in enumerate(ts):
            A, Cv, ab = self.coef(int(t))
            x = lat.float()
            if i in O and len(hist) >= 1:
                kinds['O'] += 1
                j, vj, x0j = hist[-1]
                if ford == 'reuse' or len(hist) < 2:
                    v = vj
                else:
                    j1, vj1, x0j1 = hist[-2]
                    r = odamp * (i - j) / (j - j1)
                    if 'ocoef' in plan and str(i) in plan['ocoef']:
                        r = plan['ocoef'][str(i)] * (i - j) / (j - j1)
                    if plan.get('calib'):
                        # small-training: closed-loop LS fit of forecast coefficient against the true DiT output
                        self.ex.step = i; self.ex.reuse_now = set(); self.ex.reuse_u_now = set(); self.ex.branches = ['u', 'c']
                        snap = dict(self.ex.mem)
                        uu, cc = self.dit_call(lat, [ne, pe], t).chunk(2)
                        self.ex.mem = snap  # probe must not refresh block caches
                        vt = uu + CFG * (cc - uu)
                        base = (i - j) / (j - j1)
                        if ford == 'v1':
                            dlt, tgt = base * (vj - vj1), vt - vj
                        else:
                            x0t = ab ** .5 * x - (1 - ab) ** .5 * vt
                            dlt, tgt = base * (x0j - x0j1), x0t - x0j
                        num, den = float((dlt * tgt).sum()), float((dlt * dlt).sum())
                        acc = self.calib.setdefault(i, [0., 0.]); acc[0] += num; acc[1] += den
                        r = (num / max(den, 1e-12)) * base
                    if ford == 'v1':
                        v = vj + r * (vj - vj1)
                    else:  # x0-space Taylor-1
                        x0 = x0j + r * (x0j - x0j1)
                        v = (ab ** .5 * x - x0) / (1 - ab) ** .5
                pend.append((i, v))
            else:
                self.ex.step = i
                self.ex.reuse_now = breuse.get(i, set())
                self.ex.reuse_u_now = bureuse.get(i, set())
                if i in G:
                    kinds['G'] += 1
                    self.ex.branches = ['c']
                    v = self.dit_call(lat, [pe], t)
                elif i in C and delta is not None:
                    kinds['C'] += 1
                    self.ex.branches = ['c']
                    c = self.dit_call(lat, [pe], t)
                    if cmode == 't1' and len(dhist) >= 2:
                        (j0, d0), (j1_, d1) = dhist[-2], dhist[-1]
                        u = c + d1 + cdamp * (i - j1_) / (j1_ - j0) * (d1 - d0)
                    else:
                        u = c + delta
                    v = u + CFG * (c - u)
                else:
                    kinds['F'] += 1
                    self.ex.branches = ['u', 'c']
                    u, c = self.dit_call(lat, [ne, pe], t).chunk(2)
                    delta = u - c
                    dhist = (dhist + [(i, delta)])[-2:]
                    v = u + CFG * (c - u)
                if retro_w > 0 and pend and hist:
                    j, vj, _ = hist[-1]
                    dx = torch.zeros_like(x)
                    for k, vh in pend:
                        pk = 1.
                        for mm in range(k + 1, i):
                            pk *= A_log[mm]
                        vt = vj + ((k - j) / (i - j)) * (v - vj)
                        dx += pk * A_log[('c', k)] * (vt - vh)
                    x = x + retro_w * dx
                    lat = x.half()
                pend = []
                hist = (hist + [(i, v, ab ** .5 * x - (1 - ab) ** .5 * v)])[-2:]
            A_log[i], A_log[('c', i)] = A, Cv
            lat = self.sch.step(v, t, lat, eta=0., return_dict=False)[0].half()
        torch.cuda.synchronize()
        assert torch.isfinite(lat).all()
        return lat, dict(seconds=time.perf_counter() - t0, kinds=kinds, **self.ex.calls)


def late(warm, k, lo_end=29, blocks=range(30)):
    """reuse all given blocks on steps >= warm except refresh steps (warm, warm+k, ...)."""
    return {s: set(blocks) for s in range(warm, lo_end + 1) if (s - warm) % k != 0}


def PLANS(phase):
    P = {}
    P['full'] = {}
    if phase in ('screen', 'screen2'):
        P['full_perturb1e-2'] = dict(perturb=1e-2)
        P['full_perturb1e-3'] = dict(perturb=1e-3)
    if phase == 'screen':
        for s in (20, 15, 12):
            P[f'steps{s}'] = dict(steps=s)
        # --- single components ---
        P['BR_L4'] = dict(B=late(10, 4))                       # H20 best: 1.83x
        P['BR_L3_w7'] = dict(B=late(7, 3))
        P['BR_L6_w10'] = dict(B=late(10, 6))
        P['BR_L4_t1'] = dict(B=late(10, 4), bmode='t1', damp=0.5)
        P['OP_attn_L4'] = dict(B=late(10, 4), bmode='attn')
        P['OP_attn_L2_w5'] = dict(B=late(5, 2), bmode='attn')
        P['OP_ffn_L4'] = dict(B=late(10, 4), bmode='ffn')
        P['CF_all_k2'] = dict(C=[s for s in range(2, 30) if s % 2])
        P['CF_all_k3'] = dict(C=[s for s in range(2, 30) if s % 3])
        P['CF_w10_every'] = dict(C=list(range(11, 30)))
        P['GI_last8'] = dict(G=list(range(22, 30)))
        P['GI_last12'] = dict(G=list(range(18, 30)))
        P['GI_first2'] = dict(G=[0, 1])
        P['OS_v1_k2'] = dict(O=[s for s in range(4, 30) if s % 2])
        P['OS_x01_k2'] = dict(O=[s for s in range(4, 30) if s % 2], ford='x01')
        P['OS_reuse_k2'] = dict(O=[s for s in range(4, 30) if s % 2], ford='reuse')
        P['OS_v1_k2_r25'] = dict(O=[s for s in range(4, 30) if s % 2], retro_w=0.25)
        # --- combos ---
        L4 = late(10, 4)
        refresh = [s for s in range(10, 30) if s not in L4]
        P['CB_L4+CF'] = dict(B=L4, C=[s for s in range(2, 30) if s % 2 or s in refresh])
        P['CB_L4+GI8'] = dict(B=L4, G=list(range(22, 30)))
        P['CB_L4+CF+GI8'] = dict(B=L4, C=[s for s in range(2, 22) if s % 2 or s in refresh], G=list(range(22, 30)))
        P['CB_OSx0+L4'] = dict(B={s: set(range(30)) for s in L4 if s % 2 == 0}, O=[s for s in range(4, 30) if s % 2], ford='x01')
        P['CB_attnL2+CF'] = dict(B=late(5, 2), bmode='attn', C=[s for s in range(2, 30) if s % 2])
        P['CB_steps20+L4s'] = dict(steps=20, B=late(7, 3, 19))
    if phase == 'screen2':
        # rule learned in screen: CFG-reuse (C) steps must never be block-refresh steps (else uncond block cache goes stale).
        for s in (14, 13, 11, 10):
            P[f'steps{s}'] = dict(steps=s)
        L4 = late(10, 4)
        odd = lambda lo, hi=30: [s for s in range(lo, hi) if s % 2]
        even = lambda lo, hi=30: [s for s in range(lo, hi) if s % 2 == 0]
        P['S2_L4+CFe'] = dict(B=L4, C=[3, 5, 7, 9])
        P['S2_L4+CFe+GI8'] = dict(B=L4, C=[3, 5, 7, 9], G=list(range(22, 30)))
        P['S2_L4+GI12'] = dict(B=L4, G=list(range(18, 30)))
        OSB = {s: set(range(30)) for s in L4 if s % 2 == 0}
        P['S2_OS+L4+GI8'] = dict(B=OSB, O=odd(4), ford='x01', G=even(22))
        P['S2_OS+L4+CFe'] = dict(B=OSB, O=odd(4), ford='x01', C=[4, 6, 8])
        P['S2_OS+L4+CFe+GI8'] = dict(B=OSB, O=odd(4), ford='x01', C=[4, 6, 8], G=even(22))
        P['S2_OSlate+L4'] = dict(B={s: set(range(30)) for s in L4 if s % 2 == 0}, O=odd(11), ford='x01')
        P['S2_OSlate+L4+CFe+GI8'] = dict(B={s: set(range(30)) for s in L4 if s % 2 == 0}, O=odd(11), ford='x01', C=[3, 5, 7, 9], G=even(22))
        P['S2_OSv1+L4'] = dict(B=OSB, O=odd(4), ford='v1')
        # skip 2 of every 3 steps after warmup; block reuse on alternate real late steps
        O3 = [s for s in range(4, 30) if (s - 4) % 3]
        real = [s for s in range(10, 30) if s not in O3]
        P['S2_OS3x0'] = dict(O=O3, ford='x01')
        P['S2_OS3x0+Bhalf'] = dict(O=O3, ford='x01', B={s: set(range(30)) for s in real[1::2]})
        P['S2_attnL4+OS'] = dict(B=OSB, bmode='attn', O=odd(4), ford='x01')
        P['S2_L6+OS'] = dict(B={s: set(range(30)) for s in late(10, 6) if s % 2 == 0}, O=odd(4), ford='x01')
        P['S2_attnL2w5+CFodd'] = dict(B=late(5, 2), bmode='attn', C=even(6))  # C on reuse (even) steps, refresh (odd) stay full
        P['S2_L4+CFlate'] = dict(B=L4, C=sorted(L4))  # C only on reuse steps: tests that fixed rule is sufficient
        P['S2_OS+L4_r25'] = dict(B=OSB, O=odd(4), ford='x01', retro_w=0.25)
    if phase == 'confirm':
        L4 = late(10, 4)
        P['full_perturb1e-2'] = dict(perturb=1e-2)
        P['steps16'] = dict(steps=16); P['steps12'] = dict(steps=12)
        P['BR_L4'] = dict(B=L4)
        P['BR_L6_w10'] = dict(B=late(10, 6))
        P['OP_attn_L4'] = dict(B=L4, bmode='attn')
        P['CB_L4+GI8'] = dict(B=L4, G=list(range(22, 30)))
        P['CB_L6+GI8'] = dict(B=late(10, 6), G=list(range(22, 30)))
        P['CB_OSx0+L4'] = dict(B={s: set(range(30)) for s in L4 if s % 2 == 0}, O=[s for s in range(4, 30) if s % 2], ford='x01')
        P['CB_OSx0+L4+GI'] = dict(B={s: set(range(30)) for s in L4 if s % 2 == 0}, O=[s for s in range(4, 30) if s % 2], ford='x01',
                                  G=[s for s in range(22, 30) if s % 2 == 0])
    if phase == 'screen4':
        # new components: BU (uncond-only block reuse), odamp (damped x0 forecast), cmode=t1 (Taylor CFG-delta)
        L4 = late(10, 4)
        odd = lambda lo, hi=30: [s for s in range(lo, hi) if s % 2]
        even = lambda lo, hi=30: [s for s in range(lo, hi) if s % 2 == 0]
        allb = set(range(30))
        OSB = {s: allb for s in L4 if s % 2 == 0}
        for d in (0.5, 0.75):
            P[f'S4_OSx0_d{d}'] = dict(O=odd(4), ford='x01', odamp=d)
            P[f'S4_OSx0_d{d}+L4'] = dict(B=OSB, O=odd(4), ford='x01', odamp=d)
        P['S4_OSv1_d0.5+L4'] = dict(B=OSB, O=odd(4), ford='v1', odamp=0.5)
        P['S4_CFt1_k2'] = dict(C=odd(3), cmode='t1', cdamp=0.5)
        P['S4_CFt1_k2_d1'] = dict(C=odd(3), cmode='t1', cdamp=1.0)
        P['S4_CFt1_k3'] = dict(C=[s for s in range(3, 30) if s % 3], cmode='t1', cdamp=0.5)
        P['S4_CFt1_k2_w10'] = dict(C=odd(11), cmode='t1', cdamp=0.5)
        # BU: uncond branch reuses its block residuals, cond recomputed (refresh u every k steps)
        for w, k in [(4, 2), (4, 3), (10, 2), (10, 4)]:
            P[f'S4_BU_w{w}_k{k}'] = dict(BU={s: allb for s in range(w, 30) if (s - w) % k})
        P['S4_BU_w4_k2_mid'] = dict(BU={s: set(range(3, 27)) for s in range(4, 30) if (s - 4) % 2})
        # BU in early phase + symmetric B in late phase (refresh rule: BU steps never refresh-of-B)
        P['S4_BUe+L4'] = dict(BU={s: allb for s in odd(4, 10)}, B=L4)
        P['S4_BUe+L4+GI8'] = dict(BU={s: allb for s in odd(4, 10)}, B=L4, G=list(range(22, 30)))
        # BU on B-refresh steps of L4 (only refresh cond there; u refreshed every 2nd refresh)
        ref = [s for s in range(10, 30) if s not in L4]
        P['S4_L4+BUref'] = dict(B=L4, BU={s: allb for s in ref[1::2]})
        # BU on real steps of OS stack
        realO = [s for s in range(4, 30) if s % 2 == 0]
        P['S4_OSx0+L4+BUe'] = dict(B=OSB, O=odd(4), ford='x01', BU={s: allb for s in realO if s < 10 and s > 4})
        P['S4L_OS3x0'] = dict(O=[s for s in range(4, 30) if (s - 4) % 3], ford='x01')
        P['S4_OSx0_d0.75+L4+GI'] = dict(B=OSB, O=odd(4), ford='x01', odamp=0.75, G=[s for s in even(22)])
    if phase == 'learn':
        # small-training component: per-step forecast coefficients fit on prompts 0-3 (seed 5000), tested on held-out prompts 8-15
        cf = OUT / 'calib_coef.json'
        coef = json.load(open(cf)) if cf.exists() else {}
        P['full_perturb1e-2'] = dict(perturb=1e-2)
        L4 = late(10, 4); allb = set(range(30))
        odd = lambda lo, hi=30: [s for s in range(lo, hi) if s % 2]
        OSB = {s: allb for s in L4 if s % 2 == 0}
        P['L_OSx0'] = dict(O=odd(4), ford='x01')
        P['L_OSx0+L4'] = dict(B=OSB, O=odd(4), ford='x01')
        P['L_OSx0_d0.5+L4'] = dict(B=OSB, O=odd(4), ford='x01', odamp=0.5)
        for k in ('S4_OSx0_d0.5', 'S4_OSx0_d0.5+L4', 'S4_OSv1_d0.5+L4'):
            if k in coef:
                b = PLANS('screen4')[k]; P['L_fit_' + k[3:]] = dict(b, ocoef=coef[k], odamp=1.)
        O3 = [s for s in range(4, 30) if (s - 4) % 3]
        P['L_OS3x0'] = dict(O=O3, ford='x01')
        if 'S4L_OS3x0' in coef:
            P['L_fit_OS3x0'] = dict(O=O3, ford='x01', ocoef=coef['S4L_OS3x0'])
    if phase == 'screen3':
        # push speed: stack step skip (O) + late block reuse + late guidance-off; all refresh rules respected
        allb = set(range(30))
        for warm, k in [(8, 3), (6, 3), (10, 3)]:
            O = [s for s in range(warm, 30) if (s - warm) % 2]
            real = [s for s in range(warm, 30) if s not in O]
            Bm = {s: allb for s in real if (real.index(s) % k) != 0}
            P[f'S3_O{warm}_B{k}'] = dict(O=O, ford='x01', B=Bm)
            P[f'S3_O{warm}_B{k}+GI'] = dict(O=O, ford='x01', B=Bm, G=[s for s in real if s >= 22])
        O4 = [s for s in range(4, 30) if (s - 4) % 3]
        real4 = [s for s in range(4, 30) if s not in O4]
        P['S3_OS3_Ball'] = dict(O=O4, ford='x01', B={s: allb for s in real4[2::2]})
        P['S3_OS3_Ball+GI'] = dict(O=O4, ford='x01', B={s: allb for s in real4[2::2]}, G=[s for s in real4 if s >= 22])
        # mid-only block reuse (keep first/last 3 blocks fresh) at higher density
        mid = set(range(3, 27))
        P['S3_Lmid_k6_w6'] = dict(B={s: mid for s in range(6, 30) if (s - 6) % 6})
        P['S3_Lall_k6_w6'] = dict(B={s: allb for s in range(6, 30) if (s - 6) % 6})
        P['S3_Lall_k8_w8'] = dict(B={s: allb for s in range(8, 30) if (s - 8) % 8})
        P['S3_Lall_k8_w8+GI'] = dict(B={s: allb for s in range(8, 30) if (s - 8) % 8}, G=list(range(22, 30)))
        for s in (9, 8):
            P[f'steps{s}'] = dict(steps=s)
    return P


def jsonl(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def done_set(path):
    s = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line); s.add((r['prompt'], r['seed'], r['method']))
    return s


def cmd_embed(a):
    from transformers import T5EncoderModel, T5Tokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    tok = T5Tokenizer.from_pretrained(MODEL / 'tokenizer')
    enc = T5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=torch.float16).cuda().eval()
    emb = {}
    with torch.no_grad():
        for p in ['', *TEST]:
            ids = tok(p, padding='max_length', max_length=226, truncation=True, return_tensors='pt').input_ids.cuda()
            emb[p] = enc(ids)[0].half().cpu()
    torch.save(emb, OUT / 'embeddings.pt')
    print('embeddings', len(emb))


def cmd_verify(a):
    """own loop == official pipeline (full plan)."""
    from diffusers import CogVideoXPipeline
    S = Sampler()
    p = TEST[0]
    z, info = S.run(p, 3000, {})
    class DP(CogVideoXPipeline):
        @property
        def _execution_device(self):
            return torch.device('cuda')
    for b in S.dit.transformer_blocks:  # restore original forward via executor with no reuse (already equivalent)
        pass
    S.ex.reset()
    pipe = DP(vae=None, transformer=S.dit, scheduler=S.sch, tokenizer=None, text_encoder=None)
    pipe.set_progress_bar_config(disable=True)
    init = torch.randn((1, LT, 16, 60, 90), generator=torch.Generator(device='cuda').manual_seed(3000), device='cuda', dtype=torch.float16)
    S.ex.step = -1
    zp = pipe(latents=init, prompt_embeds=S.emb[p].cuda(), negative_prompt_embeds=S.emb[''].cuda(), height=H, width=W_, num_frames=F,
              num_inference_steps=30, guidance_scale=CFG, eta=0., output_type='latent', generator=torch.Generator(device='cuda').manual_seed(3000)).frames
    d = float((z.float() - zp.float()).abs().max())
    r = dict(max_abs_diff_vs_pipeline=d, seconds_full=info['seconds'], gpu=torch.cuda.get_device_name())
    (OUT / 'verify.json').write_text(json.dumps(r, indent=1))
    print(r)
    assert d < 1e-2, d


def cmd_gen(a):
    OUT.mkdir(parents=True, exist_ok=True)
    S = Sampler()
    P = PLANS(a.phase)
    if a.only:
        P = {k: v for k, v in P.items() if k in a.only.split(',') or k == 'full'}
    L = OUT / f'gen_{a.phase}.jsonl'
    done = done_set(L)
    json.dump({k: {kk: (sorted(vv) if isinstance(vv, (set, list)) else ({str(s): sorted(b) for s, b in vv.items()} if isinstance(vv, dict) else vv))
                    for kk, vv in v.items()} for k, v in P.items()}, open(OUT / f'plans_{a.phase}.json', 'w'), indent=0)
    for i in range(a.p0, a.p0 + a.nprompt):
        for seed in a.seeds:
            d = LAT / f'p{i:02d}_s{seed}'; d.mkdir(parents=True, exist_ok=True)
            for name, plan in P.items():
                if (i, seed, name) in done:
                    continue
                if name != 'full' and not (d / 'full.pt').exists():
                    raise RuntimeError('full missing')
                z, info = S.run(TEST[i], seed, plan)
                torch.save(z.cpu(), d / f'{name}.pt')
                mse = 0. if name == 'full' else float((z.float().cpu() - torch.load(d / 'full.pt', weights_only=True).float()).square().mean())
                jsonl(L, dict(phase=a.phase, prompt=i, seed=seed, method=name, latent_mse_to_full=mse, **info))


def cmd_score(a):
    from diffusers import AutoencoderKLCogVideoX
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    L = OUT / f'score_{a.phase}.jsonl'
    done = done_set(L)
    gen = [json.loads(l) for l in open(OUT / f'gen_{a.phase}.jsonl')]
    todo = sorted({(r['prompt'], r['seed']) for r in gen})
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL / 'vae', torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
    vae.enable_tiling()
    cp = ROOT / 'weights/clip-vit-large-patch14'
    clip = CLIPModel.from_pretrained(cp, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(cp)
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
                z = torch.load(d / f'{name}.pt', weights_only=True).half().cuda()
                v = vae.decode(z.permute(0, 2, 1, 3, 4) / vae.config.scaling_factor).sample.float()
                v = ((v.clamp(-1, 1) + 1) / 2)[0].cpu()
                if name == 'full':
                    ref = v
                    if (i, s, name) in done:
                        continue
                frames = [Image.fromarray((v[:, t].permute(1, 2, 0) * 255).round().byte().numpy()) for t in range(v.shape[1])]
                imf = clip.get_image_features(pixel_values=proc(images=frames, return_tensors='pt')['pixel_values'].half().cuda())
                imf = imf / imf.norm(dim=-1, keepdim=True)
                clip_score = float((imf @ tf.T).mean() * 100)
                consist = float(((imf[1:] * imf[:-1]).sum(-1)).mean())
                gray = v.mean(0)
                lap = gray[:, 1:-1, 1:-1] * 4 - gray[:, :-2, 1:-1] - gray[:, 2:, 1:-1] - gray[:, 1:-1, :-2] - gray[:, 1:-1, 2:]
                mse = float((v - ref).square().mean())
                if i < 3 and s == min(r['seed'] for r in gen):
                    small = [f.resize((360, 240)) for f in frames]
                    small[0].save(gif / f'{a.phase}_p{i:02d}_{name}.gif', save_all=True, append_images=small[1:], duration=125, loop=0)
                jsonl(L, dict(phase=a.phase, prompt=i, seed=s, method=name, clip_score=clip_score, frame_consistency=consist,
                              motion=float((gray[1:] - gray[:-1]).abs().mean()), sharpness=float(lap.var(dim=(1, 2)).mean()),
                              psnr_to_full=None if mse == 0 else float(-10 * math.log10(mse)),
                              temporal_delta_error=float(((v[:, 1:] - v[:, :-1]) - (ref[:, 1:] - ref[:, :-1])).square().mean())))


def cmd_summary(a):
    import numpy as np
    gen = [json.loads(l) for l in open(OUT / f'gen_{a.phase}.jsonl')]
    sp = OUT / f'score_{a.phase}.jsonl'
    sc = {(r['prompt'], r['seed'], r['method']): r for r in map(json.loads, open(sp))} if sp.exists() else {}
    full_t = {(r['prompt'], r['seed']): r['seconds'] for r in gen if r['method'] == 'full'}
    fs = {k[:2]: v for k, v in sc.items() if k[2] == 'full'}
    rows = {}
    for r in gen:
        k = (r['prompt'], r['seed'])
        e = rows.setdefault(r['method'], dict(speed=[], lmse=[], psnr=[], dclip=[], sharp=[], motion=[], tde=[]))
        e['speed'].append(full_t[k] / r['seconds'])
        e['lmse'].append(r['latent_mse_to_full'])
        s = sc.get((*k, r['method']))
        if s and k in fs:
            if s['psnr_to_full'] is not None:
                e['psnr'].append(s['psnr_to_full'])
            e['dclip'].append(s['clip_score'] - fs[k]['clip_score'])
            e['sharp'].append(s['sharpness'] / fs[k]['sharpness'])
            e['motion'].append(s['motion'] / max(fs[k]['motion'], 1e-9))
            e['tde'].append(s['temporal_delta_error'])
    m = lambda x: float(np.mean(x)) if x else float('nan')
    lines = [f'# CogVideoX-2b combo screen ({a.phase}), 17f 480x720, DDIM-30, CFG6, V100. speed = paired wall-time vs full (DiT only)', '',
             '| method | n | speed | latent MSE | PSNR | dCLIP | sharp ratio | motion ratio | tdelta err |', '|---|---|---|---|---|---|---|---|---|']
    res = {}
    for k, e in sorted(rows.items(), key=lambda kv: -m(kv[1]['speed'])):
        res[k] = {kk: m(v) for kk, v in e.items()}
        lines.append(f"| {k} | {len(e['speed'])} | {m(e['speed']):.2f} | {m(e['lmse']):.4f} | {m(e['psnr']):.2f} | {m(e['dclip']):+.2f} | "
                     f"{m(e['sharp']):.3f} | {m(e['motion']):.3f} | {m(e['tde']):.5f} |")
    (OUT / f'SUMMARY_{a.phase}.md').write_text('\n'.join(lines) + '\n')
    json.dump(res, open(OUT / f'summary_{a.phase}.json', 'w'), indent=1)
    print('\n'.join(lines))


def cmd_calib(a):
    """fit per-step forecast coefficients on calibration prompts (disjoint from test prompts p0..)."""
    S = Sampler()
    base = PLANS('screen4')
    res = {}
    for name in a.only.split(','):
        plan = dict(base[name]); plan['calib'] = True
        S.calib = {}
        for i in range(a.p0, a.p0 + a.nprompt):
            for seed in a.seeds:
                S.run(TEST[i], seed, plan)
        res[name] = {str(k): v[0] / max(v[1], 1e-12) for k, v in sorted(S.calib.items())}
        print(name, res[name], flush=True)
    f = OUT / 'calib_coef.json'
    old = json.load(open(f)) if f.exists() else {}
    old.update(res); json.dump(old, open(f, 'w'), indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest='cmd', required=True)
    sp.add_parser('embed'); sp.add_parser('verify')
    for n in ('gen', 'score', 'summary', 'calib'):
        q = sp.add_parser(n); q.add_argument('--phase', default='screen')
        q.add_argument('--nprompt', type=int, default=8); q.add_argument('--p0', type=int, default=0); q.add_argument('--seeds', type=int, nargs='+', default=[3000])
        q.add_argument('--only', default='')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'run_meta.jsonl', 'a') as fh:
        fh.write(json.dumps(dict(cmd=a.cmd, args=vars(a), sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), start=time.strftime('%F %T'))) + '\n')
    dict(embed=cmd_embed, verify=cmd_verify, gen=cmd_gen, score=cmd_score, summary=cmd_summary, calib=cmd_calib)[a.cmd](a)
